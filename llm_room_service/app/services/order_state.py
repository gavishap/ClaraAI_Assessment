from enum import Enum

class OrderState(str, Enum):
    """Enum representing possible states in the order processing pipeline."""
    INITIAL = "initial"
    INTENT_CLASSIFICATION = "intent_classification"
    MENU_INQUIRY = "menu_inquiry"
    ORDER_EXTRACTION = "order_extraction"
    ITEM_VALIDATION = "item_validation"
    ITEM_SELECTION = "item_selection"
    MODIFICATION_VALIDATION = "modification_validation"
    MODIFICATION_SELECTION = "modification_selection"
    QUANTITY_VALIDATION = "quantity_validation"
    QUANTITY_ADJUSTMENT = "quantity_adjustment"
    ORDER_CONFIRMATION = "order_confirmation"
    ORDER_COMPLETED = "order_completed"
    ERROR = "error" 
