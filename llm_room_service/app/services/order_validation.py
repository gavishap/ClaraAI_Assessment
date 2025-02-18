from typing import List, Dict, Tuple, Optional, Any, Union
import json
from pydantic import ValidationError, BaseModel
from loguru import logger
from openai import OpenAI

from ..models import Order, OrderItem, OrderSchema
from ..utils.fuzzy_matching import find_best_match, find_matching_modifications
from ..config import OPENAI_CONFIG, MENU_ITEMS

class LLMValidationError(Exception):
    """Custom exception for LLM validation errors."""
    def __init__(self, message: str, raw_output: Any = None, field_errors: Dict = None):
        self.message = message
        self.raw_output = raw_output
        self.field_errors = field_errors or {}
        super().__init__(self.message)

class OrderValidator:
    def __init__(self, menu_items: Dict):
        self.menu_items = menu_items
        self.client = OpenAI(api_key=OPENAI_CONFIG["api_key"])
        self.max_retries = 3
        self.fallback_threshold = 0.7
        
    def validate_llm_output(self, raw_output: str) -> Tuple[bool, Dict, List[str]]:
        """Validate the raw LLM output for schema compliance and data validity."""
        issues = []
        
        # Step 1: Validate JSON structure
        try:
            parsed_data = json.loads(raw_output)
        except json.JSONDecodeError as e:
            issues.append(f"Invalid JSON format: {str(e)}")
            return False, None, issues

        # Step 2: Validate required fields
        required_fields = {"items"}
        missing_fields = required_fields - set(parsed_data.keys())
        if missing_fields:
            issues.append(f"Missing required fields: {', '.join(missing_fields)}")

        # Step 3: Validate items array structure
        if "items" not in parsed_data or not isinstance(parsed_data["items"], list):
            issues.append("Invalid or missing 'items' array")
            return False, parsed_data, issues

        for idx, item in enumerate(parsed_data.get("items", [])):
            item_issues = self._validate_item_fields(item, idx)
            issues.extend(item_issues)

        # Step 4: Validate room number format
        if "room_number" in parsed_data:
            room_num = parsed_data["room_number"]
            if room_num is not None and not isinstance(room_num, int):
                issues.append("Room number must be an integer or null")

        # Step 5: Check for unexpected fields
        allowed_fields = {"items", "room_number"}
        extra_fields = set(parsed_data.keys()) - allowed_fields
        if extra_fields:
            issues.append(f"Unexpected fields found: {', '.join(extra_fields)}")

        return len(issues) == 0, parsed_data, issues

    def _validate_item_fields(self, item: Dict, idx: int) -> List[str]:
        """Validate individual item fields."""
        issues = []
        required_item_fields = {"name", "quantity", "modifications"}
        
        # Check required fields
        missing_fields = required_item_fields - set(item.keys())
        if missing_fields:
            issues.append(f"Item {idx}: Missing fields: {', '.join(missing_fields)}")
            
        # Validate field types
        if "name" in item and not isinstance(item["name"], str):
            issues.append(f"Item {idx}: Name must be a string")
        if "quantity" in item and not isinstance(item["quantity"], int):
            issues.append(f"Item {idx}: Quantity must be an integer")
        if "modifications" in item and not isinstance(item["modifications"], list):
            issues.append(f"Item {idx}: Modifications must be an array")
            
        return issues

    async def handle_validation_failure(self, raw_output: str, issues: List[str], original_text: str) -> Optional[Dict]:
        """Handle validation failures with intelligent fallback strategies."""
        # Strategy 1: Attempt repair with GPT-4
        repaired_output = await self._attempt_repair(raw_output, issues)
        if repaired_output:
            is_valid, parsed_data, new_issues = self.validate_llm_output(repaired_output)
            if is_valid:
                return parsed_data

        # Strategy 2: Structured re-prompting with error context
        retry_output = await self._structured_reprompt(original_text, issues)
        if retry_output:
            is_valid, parsed_data, new_issues = self.validate_llm_output(retry_output)
            if is_valid:
                return parsed_data

        # Strategy 3: Fallback to partial extraction
        return await self._extract_partial_order(original_text)

    async def _attempt_repair(self, raw_output: str, issues: List[str]) -> Optional[str]:
        """Attempt to repair invalid LLM output using GPT-4."""
        try:
            repair_prompt = f"""You are an expert JSON repair system. Fix the following JSON output to match the required schema:

Original JSON:
{raw_output}

Validation Issues:
{chr(10).join(f"- {issue}" for issue in issues)}

Required Schema:
{{
    "room_number": number or null,
    "items": [
        {{
            "name": "exact item name from menu",
            "quantity": number,
            "modifications": ["modification1", "modification2"]
        }}
    ]
}}

Return ONLY the fixed JSON with no additional text."""

            completion = self.client.chat.completions.create(
                model=OPENAI_CONFIG["model"],
                messages=[{"role": "user", "content": repair_prompt}],
                temperature=0.0,
                response_format={ "type": "json_object" }
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in repair attempt: {str(e)}")
            return None

    async def _structured_reprompt(self, original_text: str, issues: List[str]) -> Optional[str]:
        """Re-prompt with structured error feedback."""
        try:
            reprompt = f"""You are an expert order extraction system. The previous attempt to extract an order had issues:

Original Text: "{original_text}"

Validation Issues:
{chr(10).join(f"- {issue}" for issue in issues)}

Extract the order again, paying special attention to fixing the above issues.
Ensure your response is a valid JSON object with this exact schema:
{{
    "room_number": number or null,
    "items": [
        {{
            "name": "exact item name from menu",
            "quantity": number,
            "modifications": ["modification1", "modification2"]
        }}
    ]
}}"""

            completion = self.client.chat.completions.create(
                model=OPENAI_CONFIG["model"],
                messages=[{"role": "user", "content": reprompt}],
                temperature=0.0,
                response_format={ "type": "json_object" }
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in structured reprompt: {str(e)}")
            return None

    async def _extract_partial_order(self, text: str) -> Optional[Dict]:
        """Fallback strategy to extract partial order information."""
        try:
            # Use a more permissive extraction approach
            completion = self.client.chat.completions.create(
                model=OPENAI_CONFIG["model"],
                messages=[
                    {
                        "role": "system",
                        "content": """Extract any valid order information you can find, even if incomplete.
Focus on getting at least the item names correct. If unsure about quantities, default to 1.
If unsure about modifications, leave them empty."""
                    },
                    {"role": "user", "content": text}
                ],
                temperature=0.0,
                response_format={ "type": "json_object" }
            )
            
            partial_data = json.loads(completion.choices[0].message.content)
            
            # Ensure minimum valid structure
            if "items" not in partial_data:
                partial_data["items"] = []
            if "room_number" not in partial_data:
                partial_data["room_number"] = None
                
            return partial_data
        except Exception as e:
            logger.error(f"Error in partial extraction: {str(e)}")
            return None

    def validate_order(self, order: Order, inventory: Dict[str, int]) -> Tuple[bool, List[str]]:
        """Validate an order against menu and inventory."""
        issues = []
        
        # Validate menu items
        menu_issues = self._validate_menu_items(order.items)
        issues.extend(menu_issues)
        
        # Validate inventory
        inventory_issues = self._validate_inventory(order.items, inventory)
        issues.extend(inventory_issues)
        
        # Validate room number if present
        if order.room_number is not None:
            room_issues = self._validate_room_number(order.room_number)
            issues.extend(room_issues)
            
        return len(issues) == 0, issues
        
    def _validate_menu_items(self, items: List[OrderItem]) -> List[str]:
        """Validate items against the menu."""
        issues = []
        
        for item in items:
            # Check if item exists in menu
            if item.name not in self.menu_items:
                # Try fuzzy matching
                matched_item, score = find_best_match(item.name, list(self.menu_items.keys()))
                if matched_item:
                    issues.append(
                        f"Item '{item.name}' not found. Did you mean '{matched_item}'?"
                    )
                else:
                    issues.append(f"Item '{item.name}' is not on the menu")
                continue
                
            menu_item = self.menu_items[item.name]
            
            # Check modifications
            if item.modifications:
                if not menu_item["modifications_allowed"]:
                    issues.append(f"Modifications are not allowed for {item.name}")
                else:
                    # Validate each modification
                    for mod in item.modifications:
                        if mod not in menu_item["available_modifications"]:
                            # Try fuzzy matching
                            matched_mod = find_matching_modifications(
                                mod,
                                menu_item["available_modifications"]
                            )
                            if matched_mod:
                                issues.append(
                                    f"Modification '{mod}' for {item.name} not available. "
                                    f"Available modifications: {', '.join(matched_mod)}"
                                )
                            else:
                                issues.append(
                                    f"Modification '{mod}' is not available for {item.name}"
                                )
                                
        return issues
        
    def _validate_inventory(self, items: List[OrderItem], inventory: Dict[str, int]) -> List[str]:
        """Validate items against current inventory."""
        issues = []
        
        for item in items:
            if item.name in inventory:
                if inventory[item.name] < item.quantity:
                    issues.append(
                        f"Insufficient inventory for {item.name}. "
                        f"Only {inventory[item.name]} available."
                    )
                    
        return issues
        
    def _validate_room_number(self, room_number: int) -> List[str]:
        """Validate room number."""
        issues = []
        
        if room_number < 100 or room_number > 999:
            issues.append("Room number must be between 100 and 999")
            
        return issues
        
    def suggest_alternatives(self, item_name: str) -> List[str]:
        """Suggest alternative items when requested item is unavailable."""
        if item_name not in self.menu_items:
            return []
            
        category = self.menu_items[item_name]["category"]
        alternatives = [
            name for name, item in self.menu_items.items()
            if item["category"] == category and name != item_name
        ]
        
        return alternatives[:3]  # Return top 3 alternatives

# Initialize validator at module level
order_validator = OrderValidator(menu_items=MENU_ITEMS["categories"]) 
