from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, List, Optional
from loguru import logger

from ..services.menu_loader import menu_loader
from ..services.intent_classifier import intent_classifier

router = APIRouter(prefix="/inquiries", tags=["inquiries"])

class MenuInquiry(BaseModel):
    text: str

@router.get("/menu")
async def get_menu(
    category: Optional[str] = Query(None, description="Filter by category")
):
    """Get the current menu items."""
    if category:
        items = menu_loader.get_category_items(category)
        if not items:
            raise HTTPException(
                status_code=404,
                detail=f"No items found in category: {category}"
            )
        return items
    return menu_loader.get_menu()

@router.get("/menu/available")
async def get_available_items():
    """Get menu items that are currently in stock."""
    return menu_loader.get_available_items()

@router.get("/menu/categories")
async def get_categories():
    """Get list of available menu categories."""
    menu = menu_loader.get_menu()
    categories = {item["category"] for item in menu.values()}
    return sorted(list(categories))

@router.get("/menu/items/{item_name}")
async def get_item_details(item_name: str):
    """Get detailed information about a specific menu item."""
    details = menu_loader.get_item_details(item_name)
    if not details:
        raise HTTPException(
            status_code=404,
            detail=f"Item not found: {item_name}"
        )
    return details

@router.post("/classify")
async def classify_inquiry(inquiry: MenuInquiry):
    """Classify a natural language inquiry."""
    intent, confidence = intent_classifier.classify(inquiry.text)
    return {
        "intent": intent,
        "confidence": confidence,
        "explanation": intent_classifier.get_intent_explanation(inquiry.text)
    } 
