from typing import Dict, List, Optional, Tuple
from uuid import uuid4
from loguru import logger

from ..models import Order, OrderResponse, OrderItem
from ..config import MENU_ITEMS, INVENTORY
from .intent_classifier import intent_classifier
from .order_extraction import order_extractor
from .order_validation import order_validator
from ..utils.logging import log_order_request, log_order_response, log_error

class OrderProcessor:
    def __init__(self):
        self.menu_items = MENU_ITEMS
        self.inventory = INVENTORY.copy()  # Create a copy to track inventory changes
        
    def process_order(self, text: str, room_number: int) -> Tuple[Optional[OrderResponse], List[str]]:
        """Process a natural language order request."""
        # Log the request
        log_order_request(room_number, text)
        
        # Classify intent
        intent, confidence = intent_classifier.classify(text)
        if intent != "new_order":
            error_msg = f"Unable to process request. Detected intent: {intent} (confidence: {confidence:.2%})"
            log_error("Intent Error", error_msg)
            return None, [error_msg]
            
        # Extract order details
        order = order_extractor.extract_order(text, self.menu_items)
        if not order:
            error_msg = "Failed to extract order details from the request"
            log_error("Extraction Error", error_msg)
            return None, [error_msg]
            
        # Add room number to order
        order.room_number = room_number
        
        # Validate order
        is_valid, issues = order_validator.validate_order(order, self.inventory)
        if not is_valid:
            log_error("Validation Error", "\n".join(issues))
            return None, issues
            
        # Calculate order details
        total_price = self._calculate_total_price(order)
        estimated_time = self._calculate_preparation_time(order)
        
        # Create order response
        order_response = OrderResponse(
            order_id=str(uuid4()),
            status="confirmed",
            total_price=total_price,
            estimated_time=estimated_time,
            items=order.items
        )
        
        # Update inventory
        self._update_inventory(order)
        
        # Log success
        log_order_response(
            order_response.order_id,
            order_response.status,
            order_response.total_price
        )
        
        return order_response, []
        
    def _calculate_total_price(self, order: Order) -> float:
        """Calculate the total price of the order."""
        total = 0.0
        for item in order.items:
            if item.name in self.menu_items:
                total += self.menu_items[item.name]["price"] * item.quantity
        return total
        
    def _calculate_preparation_time(self, order: Order) -> int:
        """Calculate the estimated preparation time in minutes."""
        max_time = 0
        for item in order.items:
            if item.name in self.menu_items:
                item_time = self.menu_items[item.name]["preparation_time"]
                max_time = max(max_time, item_time)
        return max_time + 5  # Add 5 minutes for order processing and delivery
        
    def _update_inventory(self, order: Order) -> None:
        """Update inventory levels after a successful order."""
        for item in order.items:
            if item.name in self.inventory:
                self.inventory[item.name] -= item.quantity
                
    def get_inventory_status(self) -> Dict[str, int]:
        """Get current inventory levels."""
        return self.inventory.copy()
        
    def reset_inventory(self) -> None:
        """Reset inventory to initial levels."""
        self.inventory = INVENTORY.copy()

# Initialize processor at module level
order_processor = OrderProcessor() 
