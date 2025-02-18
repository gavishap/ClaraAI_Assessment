from llm_room_service.app.services.order_extraction import order_extractor
from llm_room_service.app.config import MENU_ITEMS
import json

def test_order_extraction():
    """Test the order extraction system."""
    print("\n=== Order Extraction Testing ===")
    print("Type 'quit' to exit\n")
    
    while True:
        # Get user input
        text = input("\nEnter order text: ").strip()
        
        if text.lower() == 'quit':
            break
            
        if not text:
            continue
            
        # Extract order
        order = order_extractor.extract_order(text, MENU_ITEMS)
        
        # Print results
        print("\nExtracted Order:")
        if order:
            # Convert to dict for pretty printing
            order_dict = {
                "room_number": order.room_number,
                "items": [
                    {
                        "name": item.name,
                        "quantity": item.quantity,
                        "modifications": item.modifications
                    }
                    for item in order.items
                ]
            }
            print(json.dumps(order_dict, indent=2))
        else:
            print("Failed to extract order")
        print("\n" + "="*50)

if __name__ == "__main__":
    test_order_extraction() 
