from typing import Optional, Dict, List
import os
import json
from openai import OpenAI
from pydantic import ValidationError
from loguru import logger

from ..models import Order, OrderItem, OrderIntent, OrderSchema, OrderItemSchema
from ..config import OPENAI_CONFIG, MENU_ITEMS
from ..utils.fuzzy_matching import find_best_match
from .menu_embeddings import menu_embedding_service

class OrderExtractor:
    def __init__(self):
        try:
            # Initialize OpenAI client
            self.client = OpenAI(api_key=OPENAI_CONFIG["api_key"])
            logger.info("Initialized OpenAI client for order extraction")
            
            # Store last raw output for validation
            self.last_raw_output = None
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    def extract_order(self, text: str, menu_items: Dict) -> Optional[Order]:
        """Extract structured order information from text using OpenAI."""
        try:
            # First, find relevant menu items using embeddings
            relevant_items = menu_embedding_service.find_similar_items(text, threshold=0.6)
            
            # Create a focused context with only the relevant items
            menu_context = "Relevant Menu Items:\n\n"
            for item_name, score, data in relevant_items:
                category = data["category"]
                details = data["details"]
                mods = ", ".join(details["available_modifications"])
                menu_context += f"- {item_name} ({category})\n  Modifications: {mods}\n"

            # Call OpenAI with structured output
            completion = self.client.chat.completions.create(
                model=OPENAI_CONFIG["model"],
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an expert order extraction system for a room service application.
Your task is to extract orders with perfect accuracy, following a step-by-step process.

{menu_context}

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
2. For each item:
   - Find the closest matching menu item
   - If the match isn't exact, use the generic category (e.g., "salad" instead of assuming "Caesar Salad")
   - If you're not sure about a specific item, use the generic term and let validation handle it
3. Extract any quantities specified (default to 1)
4. Include ALL modifications mentioned by the user, even if they're not in the available list
5. Look for a room number (set to null if none found)

Remember:
1. NEVER omit any items mentioned in the order
2. DO NOT assume specific items when user gives generic terms (e.g., "salad" should stay as "salad")
3. Include ALL modifications mentioned by the user, even if not in the available list
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
            self.last_raw_output = response_text  # Store the raw output
            
            try:
                order_data = OrderSchema.model_validate_json(response_text)
            except ValidationError as ve:
                logger.error(f"Initial validation error: {ve}")
                # Try to recover with a reprompt
                return self._handle_extraction_failure(text, str(ve))

            # Save outputs to file for debugging
            with open("order_extraction_output.txt", "a", encoding='utf-8') as f:
                f.write("\n" + "="*50 + "\n")
                f.write("NEW EXTRACTION\n")
                f.write("="*50 + "\n")
                f.write(f"Input Text:\n{text}\n\n")
                f.write(f"Relevant Items:\n{menu_context}\n\n")
                f.write(f"Raw Response:\n{response_text}\n\n")
                f.write(f"Parsed Order:\n{order_data.model_dump_json(indent=2)}\n")
                f.write("="*50 + "\n\n")

            # Create order items directly from the extracted data
            items = [
                OrderItem(
                    name=item.name,
                    quantity=item.quantity,
                    modifications=item.modifications,
                    category="Main"  # Let validation handle the proper categorization
                )
                for item in order_data.items
            ]

            if not items:
                logger.error("No items found in the order")
                return self._handle_extraction_failure(text, "Failed to extract any items from the order")

            return Order(
                items=items,
                intent=OrderIntent.NEW_ORDER,
                room_number=order_data.room_number
            )

        except Exception as e:
            logger.error(f"Error in order extraction: {str(e)}")
            return self._handle_extraction_failure(text, str(e))

    def _handle_extraction_failure(self, text: str, error_msg: str) -> Optional[Order]:
        """Handle extraction failures by reprompting with more explicit instructions."""
        try:
            logger.info(f"Attempting to recover from extraction failure: {error_msg}")
            
            # Create a more explicit prompt that includes the error context
            completion = self.client.chat.completions.create(
                model=OPENAI_CONFIG["model"],
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an expert order extraction system. 
The previous attempt to extract an order failed with this error: {error_msg}

Please try again with these specific instructions:
1. Extract EXACTLY what the user ordered, using their exact words
2. Do not try to match or correct item names
3. Include ALL modifications exactly as mentioned
4. If unsure about a specific item, use the exact term the user used
5. Focus on accuracy over completeness

You must respond with a JSON object that follows this exact schema:
{{
    "room_number": number or null,
    "items": [
        {{
            "name": "exact item name as mentioned by user",
            "quantity": number (default to 1 if not specified),
            "modifications": ["modification1", "modification2"] (empty array if none)
        }}
    ]
}}"""
                    },
                    {
                        "role": "user",
                        "content": f"Extract the order from: {text}"
                    }
                ],
                response_format={ "type": "json_object" },
                temperature=0.0
            )
            
            response_text = completion.choices[0].message.content
            order_data = OrderSchema.model_validate_json(response_text)
            
            # Create order items directly from the extracted data
            items = [
                OrderItem(
                    name=item.name,
                    quantity=item.quantity,
                    modifications=item.modifications,
                    category="Main"  # Let validation handle the proper categorization
                )
                for item in order_data.items
            ]
            
            return Order(
                items=items,
                intent=OrderIntent.NEW_ORDER,
                room_number=order_data.room_number
            )
            
        except Exception as e:
            logger.error(f"Recovery attempt failed: {str(e)}")
            return None

# Initialize extractor at module level
order_extractor = OrderExtractor()
