from typing import Dict, List
from loguru import logger

class ResponseFormatter:
    @staticmethod
    def format_order_details(order: Dict) -> str:
        """Format order details for display."""
        output = "\nOrder Details:"
        output += "\n" + "=" * 40
        for item in order["items"]:
            mods = f" with {', '.join(item['modifications'])}" if item['modifications'] else ""
            output += f"\n- {item['quantity']}x {item['name']}{mods}"
        output += "\n" + "=" * 40
        return output

    @staticmethod
    def format_inventory_status(inventory_status: Dict) -> str:
        """Format inventory status for display."""
        output = "\nInventory Status:"
        output += "\n" + "=" * 40
        for item_name, status in inventory_status.items():
            output += f"\n- {item_name}: {status['ordered']} ordered, {status['remaining']} remaining"
        output += "\n" + "=" * 40
        return output

    @staticmethod
    def format_validation_prompts(validation: Dict) -> List[str]:
        """Format validation prompts for user interaction."""
        prompts = []
        if not validation.get("prompts"):
            return prompts

        for prompt in validation["prompts"]:
            prompts.append(f"\n{prompt}")
        return prompts

    @staticmethod
    def format_success_response(result: Dict) -> str:
        """Format successful response output."""
        output = "\n✓ "
        if result.get("type") == "inquiry":
            output += "Menu Inquiry Response:\n"
            output += result["response"]
        else:  # order
            output += "Order processed successfully!"
            if result.get("message"):
                output += f"\n{result['message']}"
            if "order" in result:
                output += ResponseFormatter.format_order_details(result["order"])
            if "inventory_status" in result:
                output += ResponseFormatter.format_inventory_status(result["inventory_status"])
        return output

    @staticmethod
    def format_error_response(result: Dict) -> str:
        """Format error response output."""
        output = "\n❌ Processing failed"
        if "error" in result:
            output += f"\n\nReason: {result['error']}"
        elif "validation" in result:
            prompts = ResponseFormatter.format_validation_prompts(result["validation"])
            output += "".join(prompts)
            if "order" in result:
                output += ResponseFormatter.format_order_details(result["order"])
        return output

    @staticmethod
    def format_response(result: Dict) -> str:
        """Format the complete response for display."""
        if result["success"]:
            return ResponseFormatter.format_success_response(result)
        else:
            return ResponseFormatter.format_error_response(result)

# Initialize formatter at module level
response_formatter = ResponseFormatter() 
