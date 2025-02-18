from typing import Optional, Dict, List
import os
import json
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from loguru import logger

from ..models import Order, OrderItem, OrderIntent
from ..config import OPENAI_CONFIG, MENU_ITEMS
from ..utils.fuzzy_matching import find_best_match

# Step 1: Define our Pydantic models for structured output
class OrderItemSchema(BaseModel):
    name: str
    quantity: int
    modifications: List[str]

class OrderSchema(BaseModel):
    room_number: Optional[int]
    items: List[OrderItemSchema]

class OrderExtractor:
    def __init__(self):
        try:
            # Initialize OpenAI client
            self.client = OpenAI(api_key=OPENAI_CONFIG["api_key"])
            logger.info("Initialized OpenAI client for order extraction")
            
            # Create menu context for the model
            self.menu_context = self._create_menu_context()
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    def _create_menu_context(self) -> str:
        """Create a formatted menu context for the model."""
        menu_text = "Available Menu Items:\n\n"
        
        for category, items in MENU_ITEMS["categories"].items():
            menu_text += f"{category}:\n"
            for item_name, details in items.items():
                mods = ", ".join(details["available_modifications"])
                menu_text += f"- {item_name} (Modifications: {mods})\n"
            menu_text += "\n"
        
        return menu_text

    def extract_order(self, text: str, menu_items: Dict) -> Optional[Order]:
        """Extract structured order information from text using OpenAI."""
        try:
            # Call OpenAI with structured output
            completion = self.client.chat.completions.create(
                model=OPENAI_CONFIG["model"],
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an expert order extraction system for a room service application.
Your task is to extract orders with perfect accuracy, following a step-by-step process.

{self.menu_context}

You must respond with a JSON object that follows this exact schema:
{{
    "room_number": number or null,
    "items": [
        {{
            "name": "exact item name from menu",
            "quantity": number (default to 1 if not specified),
            "modifications": ["modification1", "modification2"] (empty array if none)
        }}
    ]
}}

Follow these steps in your reasoning:
1. First, identify all menu items mentioned in the order
2. For each item, determine its exact name from the menu
3. Extract any quantities specified (default to 1)
4. Match any modifications to the available modifications list
5. Look for a room number (set to null if none found)

Here are some examples:

Input: "I'd like a club sandwich with extra bacon and a side of french fries"
Reasoning:
1. Found items: "club sandwich" and "french fries"
2. Menu matches: "Club Sandwich" and "French Fries"
3. No quantities specified, using default of 1
4. Modifications: "extra bacon" for Club Sandwich, none for French Fries
5. No room number mentioned
Output: {{
    "room_number": null,
    "items": [
        {{
            "name": "Club Sandwich",
            "quantity": 1,
            "modifications": ["extra bacon"]
        }},
        {{
            "name": "French Fries",
            "quantity": 1,
            "modifications": []
        }}
    ]
}}

Input: "Can I get two waters and a caesar salad with chicken to room 405"
Reasoning:
1. Found items: "waters" and "caesar salad"
2. Menu matches: "Still Water" and "Caesar Salad"
3. Quantities: 2 waters, 1 salad
4. Modifications: "add chicken" for Caesar Salad
5. Room number: 405
Output: {{
    "room_number": 405,
    "items": [
        {{
            "name": "Still Water",
            "quantity": 2,
            "modifications": []
        }},
        {{
            "name": "Caesar Salad",
            "quantity": 1,
            "modifications": ["add chicken"]
        }}
    ]
}}

Remember:
1. NEVER omit any items mentioned in the order
2. ALWAYS match item names EXACTLY as they appear in the menu
3. ONLY use modifications that are listed as available
4. Include room number if mentioned, otherwise null
5. Default quantity to 1 if not specified"""
                    },
                    {
                        "role": "user",
                        "content": f"Extract the order from this text: {text}"
                    }
                ],
                response_format={ "type": "json_object" },
                temperature=0.0,  # Set to 0 for maximum determinism
                max_tokens=OPENAI_CONFIG["max_tokens"]
            )

            # Get the response and parse it
            response_text = completion.choices[0].message.content
            order_data = OrderSchema.model_validate_json(response_text)

            # Save outputs to file for debugging
            with open("order_extraction_output.txt", "a", encoding='utf-8') as f:
                f.write("\n" + "="*50 + "\n")
                f.write("NEW EXTRACTION\n")
                f.write("="*50 + "\n")
                f.write(f"Input Text:\n{text}\n\n")
                f.write(f"Raw Response:\n{response_text}\n\n")
                f.write(f"Parsed Order:\n{order_data.model_dump_json(indent=2)}\n")
                f.write("="*50 + "\n\n")

            # Validate and clean up the order data
            items = []
            for item_data in order_data.items:
                # Search through all categories
                for category, category_items in MENU_ITEMS["categories"].items():
                    # Find best matching menu item in this category
                    menu_item, score = find_best_match(item_data.name, list(category_items.keys()))
                    if menu_item and score > 0.95:  # Increased threshold for stricter matching
                        # Validate modifications
                        valid_mods = []
                        if category_items[menu_item]["modifications_allowed"]:
                            available_mods = category_items[menu_item]["available_modifications"]
                            for mod in item_data.modifications:
                                mod_match, mod_score = find_best_match(mod, available_mods)
                                if mod_match and mod_score > 0.95:  # Increased threshold
                                    valid_mods.append(mod_match)

                        items.append(OrderItem(
                            name=menu_item,
                            quantity=item_data.quantity,
                            modifications=valid_mods,
                            category=category
                        ))
                        break

            if not items:
                logger.error("No valid items found in the order")
                return None

            return Order(
                items=items,
                intent=OrderIntent.NEW_ORDER,
                room_number=order_data.room_number
            )

        except ValidationError as ve:
            logger.error(f"Validation Error in order extraction: {str(ve)}")
            return None
        except Exception as e:
            logger.error(f"Error in order extraction: {str(e)}")
            return None

# Initialize extractor at module level
order_extractor = OrderExtractor()
