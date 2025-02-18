from pathlib import Path
from typing import Dict, Any
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# OpenAI Configuration
OPENAI_CONFIG = {
    "model": "gpt-4o-2024-08-06",  # Use the latest model that supports structured outputs
    "api_key": os.getenv("OPENAI_API_KEY"),  # Get API key from environment variable
    "temperature": float(os.getenv("MODEL_TEMPERATURE", "0.0")),  # Get from env or default to 0.0
    "max_tokens": int(os.getenv("MODEL_MAX_TOKENS", "1000")),  # Get from env or default to 1000
}

# Model configurations
INTENT_MODEL_CONFIG = {
    "primary_model": "facebook/bart-large-mnli",
    "fallback_model": "cross-encoder/nli-deberta-v3-base",
    "primary_config": {
        "max_length": 512,
        "padding": True,
        "truncation": True,
        "confidence_threshold": 0.70,
        "hypothesis_template": "This text {}",
        "score_weights": {
            "entailment": 1.0,
            "contradiction": -0.6,
            "neutral": 0.1
        },
        "intent_patterns": {
            "new_order": [
                "mentions any food or drink item",
                "contains the name of a dish or beverage",
                "specifies food items or ingredients",
                "includes any menu item or food description",
                "refers to a type of food or drink",
                "describes food preferences or choices",
                "indicates desired food or beverage items",
                "mentions modifications to food items",
                "lists any food or drink items",
                "expresses food or drink preferences"
            ],
            "general_inquiry": [
                "asks for information without mentioning specific items",
                "seeks to understand menu options or prices",
                "contains questions about food availability",
                "requests details about menu items",
                "asks about food service information",
                "inquires about dietary restrictions or options",
                "asks about ingredients or preparation methods",
                "seeks clarification about menu choices",
                "requests information about food allergies or preferences",
                "asks about available menu modifications"
            ],
            "unsupported_action": [
                "asks about an order that was already placed",
                "requests information about order status or readiness",
                "seeks to modify existing arrangements",
                "refers to a previous order or its status",
                "asks about order tracking or preparation",
                "wants to stop or cancel an existing order",
                "requests changes to food already ordered",
                "asks if ordered items are ready or complete",
                "wants to know when an order will be ready",
                "seeks to terminate or cancel current orders"
            ],
            "unknown": [
                "contains no food or menu related terms",
                "uses only vague words without food mentions",
                "has no reference to food or drinks",
                "lacks any menu or food related terms",
                "contains only general words without food context",
                "makes no mention of food or beverages",
                "uses ambiguous terms without food references",
                "provides no food or menu related information",
                "contains no recognizable food items",
                "expresses only vague preferences without food mentions"
            ]
        }
    },
    "fallback_config": {
        "max_length": 512,
        "padding": True,
        "truncation": True,
        "confidence_threshold": 0.80,
        "label_mapping": {
            "contradiction": 0,
            "entailment": 1,
            "neutral": 2
        },
        "score_weights": {
            "new_order": {
                "contradiction": -0.4,
                "entailment": 1.0,
                "neutral": 0.2
            },
            "general_inquiry": {
                "contradiction": -0.6,
                "entailment": 0.8,
                "neutral": 0.4
            },
            "unsupported_action": {
                "contradiction": -0.2,
                "entailment": 1.0,
                "neutral": 0.3
            },
            "unknown": {
                "contradiction": 0.2,
                "entailment": 0.2,
                "neutral": 1.0
            }
        },
        "intent_patterns": {
            "new_order": [
                "The text specifies new food or drink items to be delivered",
                "The message contains a list of items for a first-time order",
                "The user describes what they would like to receive now",
                "The text clearly states items for a new order",
                "The message indicates items not previously ordered"
            ],
            "general_inquiry": [
                "The text asks questions about the menu",
                "The user wants to know about options",
                "The message seeks information without ordering",
                "The text asks about food characteristics",
                "The user inquires about menu possibilities"
            ],
            "unsupported_action": [
                "The text refers to an already placed order",
                "The user asks about order status or readiness",
                "The message is about tracking an existing order",
                "The text mentions canceling or stopping orders",
                "The user wants to know if their order is ready",
                "The message checks on order preparation status",
                "The text inquires about a previously placed order",
                "The user asks when their order will be done"
            ],
            "unknown": [
                "The text is too vague to understand",
                "The user's request lacks specific details",
                "The message needs more information",
                "The text uses unclear or general terms",
                "The request is not specific enough"
            ]
        }
    },
    
    # Common rules and keywords for both models
    "keywords": {
        "menu_inquiry": ["menu", "food", "dish", "ingredient", "vegetarian", "vegan", "allergy", "spicy", "price"],
        "order_actions": ["send", "bring", "deliver", "get", "want", "order", "place", "have"],
        "menu_items": ["sandwich", "water", "pizza", "salad", "burger", "juice", "pie", "fries", "bottle"],
        "unsupported": ["cancel", "status", "ready", "check", "track", "change", "modify", "gym", "pool", "spa", 
                       "housekeeping", "clean", "towel", "wake-up", "checkout", "wifi", "internet", "parking"],
        "vague_terms": ["something good", "anything", "whatever", "something nice"]
    },
    
    # Score adjustment rules
    "score_adjustments": {
        "boost_multiplier": 1.2,
        "reduction_multiplier": 0.5,
        "min_menu_item_confidence": 0.7,
        "max_boost_score": 0.99
    }
}

ORDER_EXTRACTION_MODEL_CONFIG = {
    "model_name": "google/flan-t5-base",
    "max_input_length": 512,
    "max_output_length": 256,  # Increased to handle larger JSON responses
    "num_beams": 5,  # More structured output
    "temperature": 0.3,  # Low temperature → More deterministic
    "top_p": 0.85  # Slightly reduced to minimize hallucinations
}


# Data paths
MENU_PATH = DATA_DIR / "menu.json"
INVENTORY_PATH = DATA_DIR / "inventory.json"

# API configurations
API_CONFIG = {
    "title": "Room Service Order Management System",
    "version": "0.1.0",
    "description": "Process room service orders via natural language",
    "host": os.getenv("HOST", "0.0.0.0"),
    "port": int(os.getenv("PORT", "8000"))
}

# Validation settings
VALIDATION_CONFIG = {
    "min_room_number": 100,
    "max_room_number": 999,
    "max_special_instructions_length": 500
}

def load_menu() -> Dict[str, Any]:
    """Load menu data from JSON file."""
    with open(MENU_PATH) as f:
        return json.load(f)

def load_inventory() -> Dict[str, int]:
    """Load inventory data from JSON file."""
    with open(INVENTORY_PATH) as f:
        return json.load(f)

# Load data at module level
try:
    MENU_ITEMS = load_menu()
    INVENTORY = load_inventory()
except Exception as e:
    raise RuntimeError(f"Failed to load data files: {e}") 
