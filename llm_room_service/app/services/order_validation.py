from typing import List, Dict, Tuple
from loguru import logger

from ..models import Order, OrderItem
from ..utils.fuzzy_matching import find_best_match, find_matching_modifications

class OrderValidator:
    def __init__(self, menu_items: Dict):
        self.menu_items = menu_items
        
    def validate_order(self, order: Order, inventory: Dict[str, int]) -> Tuple[bool, List[str]]:
        """Validate an order against menu and inventory."""
        issues = []
        
        # Validate menu items
        menu_issues = self._validate_menu_items(order.items)
        issues.extend(menu_issues)
        
        # Validate inventory
        inventory_issues = self._validate_inventory(order.items, inventory)
        issues.extend(inventory_issues)
        
        # Validate room number if present
        if order.room_number is not None:
            room_issues = self._validate_room_number(order.room_number)
            issues.extend(room_issues)
            
        return len(issues) == 0, issues
        
    def _validate_menu_items(self, items: List[OrderItem]) -> List[str]:
        """Validate items against the menu."""
        issues = []
        
        for item in items:
            # Check if item exists in menu
            if item.name not in self.menu_items:
                # Try fuzzy matching
                matched_item, score = find_best_match(item.name, list(self.menu_items.keys()))
                if matched_item:
                    issues.append(
                        f"Item '{item.name}' not found. Did you mean '{matched_item}'?"
                    )
                else:
                    issues.append(f"Item '{item.name}' is not on the menu")
                continue
                
            menu_item = self.menu_items[item.name]
            
            # Check modifications
            if item.modifications:
                if not menu_item["modifications_allowed"]:
                    issues.append(f"Modifications are not allowed for {item.name}")
                else:
                    # Validate each modification
                    for mod in item.modifications:
                        if mod not in menu_item["available_modifications"]:
                            # Try fuzzy matching
                            matched_mod = find_matching_modifications(
                                mod,
                                menu_item["available_modifications"]
                            )
                            if matched_mod:
                                issues.append(
                                    f"Modification '{mod}' for {item.name} not available. "
                                    f"Available modifications: {', '.join(matched_mod)}"
                                )
                            else:
                                issues.append(
                                    f"Modification '{mod}' is not available for {item.name}"
                                )
                                
        return issues
        
    def _validate_inventory(self, items: List[OrderItem], inventory: Dict[str, int]) -> List[str]:
        """Validate items against current inventory."""
        issues = []
        
        for item in items:
            if item.name in inventory:
                if inventory[item.name] < item.quantity:
                    issues.append(
                        f"Insufficient inventory for {item.name}. "
                        f"Only {inventory[item.name]} available."
                    )
                    
        return issues
        
    def _validate_room_number(self, room_number: int) -> List[str]:
        """Validate room number."""
        issues = []
        
        if room_number < 100 or room_number > 999:
            issues.append("Room number must be between 100 and 999")
            
        return issues
        
    def suggest_alternatives(self, item_name: str) -> List[str]:
        """Suggest alternative items when requested item is unavailable."""
        if item_name not in self.menu_items:
            return []
            
        category = self.menu_items[item_name]["category"]
        alternatives = [
            name for name, item in self.menu_items.items()
            if item["category"] == category and name != item_name
        ]
        
        return alternatives[:3]  # Return top 3 alternatives

# Initialize validator at module level
order_validator = OrderValidator(menu_items={}) 
