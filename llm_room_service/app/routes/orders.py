from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from loguru import logger

from ..models import OrderResponse
from ..services.order_processing import order_processor
from ..services.menu_loader import menu_loader
from ..utils.logging import log_error

router = APIRouter(prefix="/orders", tags=["orders"])

class OrderRequest(BaseModel):
    text: str
    room_number: int

@router.post("/", response_model=OrderResponse)
async def create_order(request: OrderRequest):
    """Process a natural language room service order."""
    logger.info(f"Received order request for room {request.room_number}")
    
    # Process the order
    response, issues = order_processor.process_order(request.text, request.room_number)
    
    if issues:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Failed to process order",
                "issues": issues
            }
        )
        
    logger.info(f"Order processed successfully: {response.order_id}")
    return response

@router.get("/status/{order_id}")
async def get_order_status(order_id: str):
    """Get the status of an existing order."""
    # Note: In a real implementation, this would query a database
    raise HTTPException(
        status_code=501,
        detail="Order status tracking not implemented"
    )

@router.get("/history")
async def get_order_history(
    room_number: Optional[int] = Query(None, description="Filter by room number"),
    limit: int = Query(10, ge=1, le=100, description="Number of orders to return")
):
    """Get order history."""
    # Note: In a real implementation, this would query a database
    raise HTTPException(
        status_code=501,
        detail="Order history not implemented"
    )

@router.post("/{order_id}/cancel")
async def cancel_order(order_id: str):
    """Cancel an existing order."""
    # Note: In a real implementation, this would update the order status
    raise HTTPException(
        status_code=501,
        detail="Order cancellation not implemented"
    ) 
