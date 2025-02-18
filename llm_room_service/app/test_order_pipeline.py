import asyncio
import json
from typing import Dict, List, Optional
from loguru import logger

from .services.order_extraction import OrderExtractor
from .services.enhanced_validation import enhanced_validator
from .services.context_manager import context_manager
from .config import MENU_ITEMS

async def process_order_text(text: str) -> Dict:
    """Process natural language order text through extraction and validation."""
    # Start new context for this order
    context = context_manager.start_new_context(text)
    logger.info(f"Started new order context for: {text}")

    # Extract order
    extractor = OrderExtractor()
    order = extractor.extract_order(text, MENU_ITEMS)
    
    if not order:
        logger.error("Failed to extract order")
        return {
            "success": False,
            "error": "Failed to extract order from text"
        }

    logger.info("✓ Order extracted successfully")
    logger.info(f"Extracted order: {order.model_dump_json(indent=2)}")
    
    # Update context with extracted order
    context_manager.update_current_context(
        extracted_order=order.model_dump()
    )

    # Load inventory (in real app, this would come from a database)
    with open("llm_room_service/data/inventory.json", "r") as f:
        inventory = json.load(f)

    # Validate order
    validation_result = enhanced_validator.validate_order(order, inventory)
    
    if validation_result.is_valid:
        logger.info("✓ Order validation passed")
        return {
            "success": True,
            "order": order.model_dump(),
            "validation": {
                "passed": True,
                "suggestions": validation_result.suggestions
            }
        }
    else:
        logger.warning("❌ Order validation failed")
        logger.warning(f"Issues: {validation_result.issues}")
        return {
            "success": False,
            "order": order.model_dump(),
            "validation": {
                "passed": False,
                "issues": validation_result.issues,
                "suggestions": validation_result.suggestions,
                "requires_user_input": validation_result.requires_user_input,
                "user_queries": validation_result.user_queries
            }
        }

async def test_order_pipeline():
    """Interactive test for order processing pipeline."""
    print("\n=== Room Service Order Processing Test ===")
    print("Enter your order in natural language (or 'quit' to exit):")
    
    while True:
        text = input("\nYour order: ").strip()
        if text.lower() == 'quit':
            break
            
        print("\nProcessing order...")
        result = await process_order_text(text)
        
        if result["success"]:
            print("\n✓ Order processed successfully!")
            if result["validation"]["suggestions"]:
                print("\nSuggestions:")
                for item, suggestions in result["validation"]["suggestions"].items():
                    print(f"- Did you mean '{suggestions[0]}' instead of '{item}'?")
        else:
            print("\n❌ Order processing failed")
            print("\nIssues:")
            for issue in result["validation"]["issues"]:
                print(f"- {issue}")
                
            if result["validation"]["requires_user_input"]:
                print("\nUser input required:")
                for query in result["validation"]["user_queries"]:
                    if query["type"] == "item_replacement":
                        print(f"\nItem '{query['item']}' not found. Did you mean one of these?")
                        for i, option in enumerate(query["options"], 1):
                            print(f"{i}. {option}")
                    elif query["type"] == "modification_removal":
                        print(f"\nModification '{query['item']}' is not valid.")
                        if query["options"]:
                            print("Available modifications:")
                            for i, option in enumerate(query["options"], 1):
                                print(f"{i}. {option}")
                    elif query["type"] == "quantity_adjustment":
                        print(f"\nInsufficient inventory for '{query['item']}'")
                        print(f"Maximum available: {query['options'][0]}")

if __name__ == "__main__":
    asyncio.run(test_order_pipeline()) 
