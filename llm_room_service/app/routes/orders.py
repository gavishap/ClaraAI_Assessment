from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from loguru import logger

from ..models import OrderResponse
from ..services.order_processing import order_processor
from ..services.menu_loader import menu_loader
from ..services.state_machine import state_machine, OrderState
from ..utils.logging import log_error

router = APIRouter(prefix="/orders", tags=["orders"])

class OrderRequest(BaseModel):
    text: str
    room_number: int

@router.post("/", response_model=OrderResponse)
async def create_order(request: OrderRequest):
    """Process a natural language room service order."""
    logger.info(f"Received order request for room {request.room_number}")
    
    # Initialize state machine if in initial state
    if state_machine.get_current_state() == OrderState.INITIAL:
        state_machine.start_new_order(request.text)
    
    try:
        # Process the order
        response, issues = order_processor.process_order(request.text, request.room_number)
        
        if issues:
            # Transition to error state
            state_machine.transition_to(
                OrderState.ERROR,
                "Order processing failed",
                {"error": issues[0]}
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Failed to process order",
                    "issues": issues
                }
            )
            
        # Transition to completed state
        state_machine.transition_to(
            OrderState.ORDER_COMPLETED,
            "Order processed successfully",
            {"order_id": response.order_id}
        )
        
        logger.info(f"Order processed successfully: {response.order_id}")
        return response
    except Exception as e:
        # Handle unexpected errors
        state_machine.transition_to(
            OrderState.ERROR,
            "Unexpected error",
            {"error": str(e)}
        )
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("/status/{order_id}")
async def get_order_status(order_id: str):
    """Get the status of an existing order."""
    # Get current state from state machine
    current_state = state_machine.get_current_state()
    
    # Return current state information
    return {
        "order_id": order_id,
        "state": current_state,
        "context": state_machine.get_context()
    }

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
    try:
        # Transition to cancelled state
        state_machine.transition_to(
            OrderState.ERROR,
            "Order cancelled by user",
            {"order_id": order_id}
        )
        return {"status": "cancelled", "order_id": order_id}
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to cancel order: {str(e)}"
        ) 
