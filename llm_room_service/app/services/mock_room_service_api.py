from typing import Dict, Optional
import uuid
from datetime import datetime
from loguru import logger

class MockRoomServiceAPI:
    def __init__(self):
        """Initialize the mock room service API."""
        self.orders = {}  # Store orders in memory
        self.kitchen_queue = []  # Simulate kitchen queue
        
    async def place_order(self, order: Dict) -> Dict:
        """Place an order with the kitchen."""
        try:
            # Generate a unique order ID
            order_id = str(uuid.uuid4())
            
            # Add timestamp and status
            order_details = {
                "order_id": order_id,
                "timestamp": datetime.now().isoformat(),
                "status": "confirmed",
                "estimated_delivery": datetime.now().isoformat(),  # In real system, would calculate based on items
                "items": order["items"],
                "room_number": order["room_number"],
                "special_instructions": order.get("special_instructions")
            }
            
            # Store the order
            self.orders[order_id] = order_details
            self.kitchen_queue.append(order_id)
            
            logger.info(f"Order placed successfully - ID: {order_id}")
            logger.info(f"Order details: {order_details}")
            
            return {
                "success": True,
                "order_id": order_id,
                "message": "Order placed successfully",
                "details": order_details
            }
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {
                "success": False,
                "error": f"Failed to place order: {str(e)}"
            }
            
    async def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get the status of an order."""
        return self.orders.get(order_id)
        
    async def update_order_status(self, order_id: str, status: str) -> bool:
        """Update the status of an order."""
        if order_id in self.orders:
            self.orders[order_id]["status"] = status
            return True
        return False
        
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order if it hasn't been prepared yet."""
        if order_id in self.orders and self.orders[order_id]["status"] == "confirmed":
            self.orders[order_id]["status"] = "cancelled"
            if order_id in self.kitchen_queue:
                self.kitchen_queue.remove(order_id)
            return True
        return False

# Initialize mock API at module level
mock_room_service_api = MockRoomServiceAPI() 
