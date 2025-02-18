import asyncio
from typing import Dict, Optional
import json
from loguru import logger

from llm_room_service.app.services.order_extraction import order_extractor
from llm_room_service.app.services.order_validation import order_validator
from llm_room_service.app.config import MENU_ITEMS
from llm_room_service.app.models import Order

async def process_order_text(text: str) -> Optional[Dict]:
    """Process natural language order text through extraction and validation."""
    print("\nProcessing order...")
    print("="*50)
    print(f"Input text: {text}")
    print("="*50)

    # Step 1: Extract order
    print("\n1. Extracting order...")
    order = order_extractor.extract_order(text, MENU_ITEMS)
    
    if not order:
        print("❌ Failed to extract order")
        return None
    
    print("✓ Order extracted successfully")
    print("\nExtracted order:")
    print(json.dumps({
        "room_number": order.room_number,
        "items": [
            {
                "name": item.name,
                "quantity": item.quantity,
                "modifications": item.modifications,
                "category": item.category
            } for item in order.items
        ]
    }, indent=2))

    # Step 2: Validate raw LLM output
    print("\n2. Validating LLM output...")
    raw_output = order_extractor.last_raw_output  # We need to add this to OrderExtractor
    is_valid, parsed_data, issues = order_validator.validate_llm_output(raw_output)
    
    if not is_valid:
        print("❌ LLM output validation failed")
        print("\nValidation issues:")
        for issue in issues:
            print(f"- {issue}")
        
        print("\nAttempting to recover...")
        fixed_data = await order_validator.handle_validation_failure(raw_output, issues, text)
        if not fixed_data:
            print("❌ Unable to recover from validation failure")
            return None
        print("✓ Successfully recovered from validation issues")
        parsed_data = fixed_data

    # Step 3: Validate business rules
    print("\n3. Validating business rules...")
    is_valid, issues = order_validator.validate_order(order, MENU_ITEMS["categories"])
    
    if not is_valid:
        print("❌ Business validation failed")
        print("\nValidation issues:")
        for issue in issues:
            print(f"- {issue}")
        return None
    
    print("✓ Order passed all validations")
    
    return parsed_data

def test_order_pipeline():
    """Interactive testing of the order pipeline."""
    print("\n=== Order Processing Pipeline Testing ===")
    print("Type 'quit' to exit\n")
    
    while True:
        # Get user input
        text = input("\nEnter order text: ").strip()
        
        if text.lower() == 'quit':
            break
            
        if not text:
            continue
        
        # Process the order
        result = asyncio.run(process_order_text(text))
        
        if result:
            print("\nFinal processed order:")
            print(json.dumps(result, indent=2))
        
        print("\n" + "="*50)

if __name__ == "__main__":
    test_order_pipeline() 
