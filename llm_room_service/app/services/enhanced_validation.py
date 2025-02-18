from typing import Dict, List, Tuple, Optional
from loguru import logger
from datetime import datetime
from pydantic import BaseModel

from ..models import Order, OrderItem
from ..config import MENU_ITEMS
from ..utils.fuzzy_matching import find_best_match
from .menu_embeddings import menu_embedding_service
from .langchain_context import langchain_context

class ValidationStep(BaseModel):
    type: str
    details: Dict
    timestamp: str

class ValidationResult(BaseModel):
    """Validation result with all necessary information."""
    is_valid: bool = True
    issues: List[str] = []
    suggestions: Dict[str, List[Tuple[str, float]]] = {}
    requires_user_input: bool = False
    user_queries: List[Dict] = []
    validation_steps: List[ValidationStep] = []

    def add_validation_step(self, step_type: str, details: Dict):
        """Add a validation step with details."""
        self.validation_steps.append(ValidationStep(
            type=step_type,
            details=details,
            timestamp=datetime.now().isoformat()
        ))

    def add_issue(self, issue: str):
        self.issues.append(issue)
        self.is_valid = False
        logger.warning(f"Validation issue: {issue}")

    def add_suggestion(self, item: str, suggestions: List[Tuple[str, float]]):
        self.suggestions[item] = suggestions
        logger.info(f"Added suggestions for '{item}': {suggestions}")

    def add_user_query(self, query_type: str, item: str, suggestions: List[Tuple[str, float]]):
        self.requires_user_input = True
        self.user_queries.append({
            "type": query_type,
            "item": item,
            "suggestions": suggestions
        })
        logger.info(f"Added user query - Type: {query_type}, Item: {item}, Suggestions: {suggestions}")

class EnhancedValidator:
    def __init__(self, menu_items: Dict):
        self.menu_items = menu_items
        # Create O(1) lookup dictionaries
        self.item_dict = self._create_item_dict()
        self.modification_dict = self._create_modification_dict()

    def _create_item_dict(self) -> Dict[str, Tuple[str, Dict]]:
        """Create O(1) lookup dictionary for menu items."""
        item_dict = {}
        for category, items in self.menu_items["categories"].items():
            for item_name, details in items.items():
                item_dict[item_name.lower()] = (category, details)
        return item_dict

    def _create_modification_dict(self) -> Dict[str, Dict[str, List[str]]]:
        """Create O(1) lookup dictionary for modifications."""
        mod_dict = {}
        for category, items in self.menu_items["categories"].items():
            for item_name, details in items.items():
                if details["modifications_allowed"]:
                    for mod in details["available_modifications"]:
                        if mod.lower() not in mod_dict:
                            mod_dict[mod.lower()] = {"items": [], "original": mod}
                        mod_dict[mod.lower()]["items"].append(item_name)
        return mod_dict

    def validate_item(self, item: OrderItem) -> ValidationResult:
        """Validate a single order item with multiple strategies."""
        result = ValidationResult()
        logger.info(f"\n{'='*50}\nStarting validation for item: {item.name}\n{'='*50}")

        # Step 1: O(1) dictionary lookup
        logger.info("\n1. Performing O(1) dictionary lookup...")
        item_key = item.name.lower()
        if item_key in self.item_dict:
            logger.info(f"✓ Success: Item '{item.name}' found in menu (direct match)")
            result.add_validation_step("dictionary_lookup", {
                "item": item.name,
                "status": "success",
                "match_type": "direct"
            })
            category, details = self.item_dict[item_key]
            return self._validate_modifications(item, details, result)

        logger.info("✗ Item not found in dictionary, proceeding to fuzzy matching")
        result.add_validation_step("dictionary_lookup", {
            "item": item.name,
            "status": "failed"
        })

        # Step 2: Fuzzy matching
        logger.info("\n2. Attempting fuzzy matching...")
        menu_items = list(self.item_dict.keys())
        matched_item, score = find_best_match(item.name, menu_items)
        
        if matched_item and score > 0.8:
            logger.info(f"✓ Success: Item '{item.name}' matched to '{matched_item}' (fuzzy match, score: {score:.2f})")
            result.add_validation_step("fuzzy_matching", {
                "item": item.name,
                "matched_to": matched_item,
                "score": score,
                "status": "success"
            })
            category, details = self.item_dict[matched_item.lower()]
            result.add_suggestion(item.name, [(matched_item, score)])
            return self._validate_modifications(item, details, result)

        logger.info("✗ No good fuzzy matches found, proceeding to embedding similarity")
        result.add_validation_step("fuzzy_matching", {
            "item": item.name,
            "status": "failed",
            "best_score": score if matched_item else 0.0
        })

        # Step 3: Embedding similarity
        logger.info("\n3. Performing embedding similarity search...")
        similar_items = menu_embedding_service.find_similar_items(item.name)

        if similar_items:
            logger.info(f"✓ Success: Found {len(similar_items)} similar items using embeddings")
            # Only log names and scores for top matches
            top_matches = [(name, score) for name, score, _ in similar_items[:2]]
            logger.info(f"Top matches: {top_matches}")
            result.add_validation_step("embedding_similarity", {
                "item": item.name,
                "status": "success",
                "matches": top_matches
            })
            suggestions = top_matches
            result.add_suggestion(item.name, suggestions)
            result.add_user_query(
                "item_replacement",
                item.name,
                suggestions
            )
            # Add removal option
            result.add_user_query(
                "item_removal",
                item.name,
                []
            )
        else:
            logger.warning(f"✗ No similar items found for '{item.name}' using any method")
            result.add_validation_step("embedding_similarity", {
                "item": item.name,
                "status": "failed"
            })
            result.add_issue(f"Item '{item.name}' not found in menu")
            result.add_user_query(
                "item_removal",
                item.name,
                []
            )

        return result

    def _validate_modifications(self, item: OrderItem, item_details: Dict, result: ValidationResult) -> ValidationResult:
        """Validate modifications for an item."""
        logger.info(f"\nValidating modifications for {item.name}...")
        
        if not item.modifications:
            logger.info("No modifications to validate")
            return result

        if not item_details["modifications_allowed"]:
            logger.warning(f"❌ Modifications are not allowed for {item.name}")
            result.add_issue(f"Modifications are not allowed for {item.name}")
            result.add_user_query(
                "modification_removal_all",
                item.name,
                []
            )
            return result

        available_mods = {mod.lower(): mod for mod in item_details["available_modifications"]}
        logger.info(f"Available modifications: {available_mods}")
        
        for mod in item.modifications:
            mod_key = mod.lower()
            logger.info(f"\nChecking modification: '{mod}'")
            
            # Step 1: O(1) dictionary lookup
            if mod_key in available_mods:
                logger.info(f"✓ Success: Modification '{mod}' is valid (direct match)")
                result.add_validation_step("modification_lookup", {
                    "modification": mod,
                    "status": "success",
                    "match_type": "direct"
                })
                continue

            # Step 2: Fuzzy matching
            matched_mod, score = find_best_match(mod, list(available_mods.values()))
            if matched_mod and score > 0.8:
                logger.info(f"✓ Success: Modification '{mod}' matched to '{matched_mod}' (fuzzy match, score: {score:.2f})")
                result.add_validation_step("modification_fuzzy", {
                    "modification": mod,
                    "matched_to": matched_mod,
                    "score": score,
                    "status": "success"
                })
                result.add_suggestion(mod, [(matched_mod, score)])
                continue

            # Step 3: Embedding similarity
            logger.info("\n3. Checking embedding similarity for modifications...")
            similar_mods = menu_embedding_service.find_similar_modifications(mod, item.name)
            
            if similar_mods:
                logger.info(f"Found similar modifications: {similar_mods}")
                result.add_suggestion(mod, similar_mods)
                result.add_user_query(
                    "modification_replacement",
                    mod,
                    similar_mods
                )
                result.add_issue(
                    f"Modification '{mod}' not available for {item.name}. "
                    "Would you like to replace it with a similar modification or remove it?"
                )
            else:
                logger.warning(f"❌ No similar modifications found for '{mod}'")
                result.add_issue(f"No similar modifications found for '{mod}'. Would you like to remove it?")
                result.add_user_query(
                    "modification_removal",
                    mod,
                    []
                )

        return result

    def validate_order(self, order: Order, inventory: Dict[str, int]) -> ValidationResult:
        """Validate entire order."""
        result = ValidationResult()
        logger.info("\n=== Starting Order Validation ===")

        # Validate each item
        for item in order.items:
            item_result = self.validate_item(item)
            result.issues.extend(item_result.issues)
            result.suggestions.update(item_result.suggestions)
            result.user_queries.extend(item_result.user_queries)
            result.requires_user_input |= item_result.requires_user_input
            result.validation_steps.extend(item_result.validation_steps)

        # Validate inventory levels
        logger.info("\n=== Validating Inventory Levels ===")
        self._validate_inventory(order, inventory, result)

        # Log validation summary
        logger.info("\n=== Validation Summary ===")
        logger.info(f"Valid: {result.is_valid}")
        logger.info(f"Issues: {len(result.issues)}")
        logger.info(f"Suggestions: {len(result.suggestions)}")
        logger.info(f"Requires user input: {result.requires_user_input}")

        # Update context with validation results
        for issue in result.issues:
            langchain_context.update_order_memory(
                validation_issue={"message": issue}
            )
        
        for item, suggestions in result.suggestions.items():
            langchain_context.update_order_memory(
                suggestion={"item": item, "suggestions": suggestions}
            )

        return result

    def _validate_inventory(self, order: Order, inventory: Dict[str, int], result: ValidationResult):
        """Validate inventory levels."""
        for item in order.items:
            # Find the item in the correct category
            for category, items in inventory.items():
                if item.name in items:
                    available_quantity = items[item.name]
                    if available_quantity < item.quantity:
                        result.add_issue(
                            f"Insufficient inventory for {item.name}. "
                            f"Only {available_quantity} available."
                        )
                        if available_quantity > 0:
                            # Suggest quantity adjustment if some stock is available
                            result.add_user_query(
                                "quantity_adjustment",
                                item.name,
                                [(str(available_quantity), 1.0)]
                            )
                        else:
                            # If no stock, suggest removal or replacement
                            result.add_user_query(
                                "item_removal",
                                item.name,
                                []
                            )
                            # Find similar items in the same category that are in stock
                            alternatives = [
                                (name, 1.0) for name, qty in items.items()
                                if qty > 0 and name != item.name
                            ][:3]  # Get top 3 alternatives
                            if alternatives:
                                result.add_user_query(
                                    "item_replacement",
                                    item.name,
                                    alternatives
                                )
                        result.is_valid = False
                    break

# Initialize validator at module level
enhanced_validator = EnhancedValidator(menu_items=MENU_ITEMS) 
