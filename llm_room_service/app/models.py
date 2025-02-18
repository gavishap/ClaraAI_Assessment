from typing import List, Optional, Dict
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid

# Add the schema classes before the other model definitions
class OrderItemSchema(BaseModel):
    """Schema for validating LLM output for order items."""
    name: str
    quantity: int
    modifications: List[str]

class OrderSchema(BaseModel):
    """Schema for validating LLM output for orders."""
    room_number: Optional[int]
    items: List[OrderItemSchema]

# Define all possible intents that can be classified
class OrderIntent(str, Enum):
    NEW_ORDER = "new_order"  # User wants to place a new order
    GENERAL_INQUIRY = "general_inquiry"  # User is asking about menu/service
    UNSUPPORTED_ACTION = "unsupported_action"  # User requests something we can't do
    UNKNOWN = "unknown"  # Can't determine the intent

class FoodCategory(str, Enum):
    MAIN = "Main"
    BEVERAGE = "Beverage"
    DESSERT = "Dessert"
    SIDE = "Side"

class MenuItem(BaseModel):
    price: float = Field(gt=0, le=999.99, description="Price of the item (max $999.99)")
    description: str = Field(..., description="Description of the item")
    modifications_allowed: bool = Field(default=False, description="Whether modifications are allowed")
    available_modifications: List[str] = Field(default_factory=list, description="List of available modifications")
    allergens: List[str] = Field(default_factory=list, description="List of allergens")
    preparation_time: int = Field(gt=0, le=180, description="Preparation time in minutes (max 180)")

    @validator("price")
    def validate_price(cls, v):
        if v <= 0 or v > 999.99:
            raise ValueError("Price must be between $0.01 and $999.99")
        return round(v, 2)

    @validator("preparation_time")
    def validate_prep_time(cls, v):
        if v <= 0 or v > 180:
            raise ValueError("Preparation time must be between 1 and 180 minutes")
        return v

class MenuCategory(BaseModel):
    items: Dict[str, MenuItem]

class Menu(BaseModel):
    categories: Dict[FoodCategory, Dict[str, MenuItem]]

class InventoryCategory(BaseModel):
    items: Dict[str, int]

class Inventory(BaseModel):
    categories: Dict[FoodCategory, Dict[str, int]]

    @validator("categories")
    def validate_inventory_levels(cls, v):
        for category, items in v.items():
            for item, quantity in items.items():
                if quantity < 0:
                    raise ValueError(f"Inventory level for {item} cannot be negative")
        return v

class OrderItem(BaseModel):
    name: str = Field(..., description="Name of the menu item")
    quantity: int = Field(gt=0, description="Quantity must be greater than zero")
    modifications: List[str] = Field(default_factory=list, description="List of modifications to the item")
    category: FoodCategory = Field(..., description="Category of the item")

    @validator("quantity")
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError("Quantity must be greater than zero")
        return v

class Order(BaseModel):
    intent: OrderIntent = Field(..., description="The intent of the user's request")
    items: List[OrderItem] = Field(..., description="List of items in the order")
    room_number: Optional[int] = Field(None, description="Room number for delivery")
    special_instructions: Optional[str] = Field(None, description="Special instructions for the order")

    @validator("room_number")
    def validate_room_number(cls, v):
        if v is not None and (v < 100 or v > 999):
            raise ValueError("Room number must be between 100 and 999")
        return v

    @validator("special_instructions")
    def validate_instructions(cls, v):
        if v and len(v) > 500:
            raise ValueError("Special instructions must be under 500 characters")
        return v

class OrderResponse(BaseModel):
    order_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the order")
    status: str = Field(..., description="Status of the order")
    total_price: Optional[float] = Field(None, description="Total price of the order")
    estimated_time: int = Field(..., description="Estimated preparation time in minutes")
    items: List[OrderItem] = Field(..., description="List of items in the order")

# Additional models for API requests/responses
class OrderRequest(BaseModel):
    text: str = Field(..., description="Natural language order text")
    room_number: int = Field(..., description="Room number for delivery")

class MenuInquiry(BaseModel):
    text: str = Field(..., description="Natural language inquiry text")
    category: Optional[FoodCategory] = Field(None, description="Optional category filter")

# Model for returning intent classification results
class IntentClassificationResponse(BaseModel):
    intent: OrderIntent  # The classified intent
    confidence: float = Field(..., description="Confidence score of the classification")
    explanation: str = Field(
        ..., 
        description="Human-readable explanation of the classification"
    )

    @validator("explanation", pre=True, always=True)
    def generate_explanation(cls, v, values):
        """Auto-generate explanation based on the classified intent."""
        intent = values.get("intent", OrderIntent.UNKNOWN)

        explanations = {
            OrderIntent.NEW_ORDER: "This appears to be a new food order request.",
            OrderIntent.GENERAL_INQUIRY: "This seems to be a general question about our menu or service.",
            OrderIntent.UNSUPPORTED_ACTION: "This request contains an action we don't support.",
            OrderIntent.UNKNOWN: "This request is ambiguous or does not match any category. Could you please clarify?"
        }
        
        return explanations.get(intent, "Unknown request type.")

