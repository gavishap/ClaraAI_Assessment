from typing import Dict, Any
import json
from pathlib import Path
from loguru import logger

from ..config import MENU_PATH, INVENTORY_PATH

class MenuLoader:
    def __init__(self):
        self._menu_items = None
        self._inventory = None
        self._load_data()
        
    def _load_data(self) -> None:
        """Load menu and inventory data from JSON files."""
        try:
            with open(MENU_PATH) as f:
                self._menu_items = json.load(f)
            with open(INVENTORY_PATH) as f:
                self._inventory = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load menu data: {e}")
            raise RuntimeError("Failed to initialize menu data")
            
    def get_menu(self) -> Dict[str, Any]:
        """Get the current menu items."""
        return self._menu_items.copy()
        
    def get_inventory(self) -> Dict[str, int]:
        """Get the current inventory levels."""
        return self._inventory.copy()
        
    def get_category_items(self, category: str) -> Dict[str, Any]:
        """Get menu items by category."""
        return {
            name: item for name, item in self._menu_items.items()
            if item["category"].lower() == category.lower()
        }
        
    def get_available_items(self) -> Dict[str, Any]:
        """Get menu items that are currently in stock."""
        return {
            name: item for name, item in self._menu_items.items()
            if self._inventory.get(name, 0) > 0
        }
        
    def get_item_details(self, item_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific menu item."""
        if item_name in self._menu_items:
            details = self._menu_items[item_name].copy()
            details["in_stock"] = self._inventory.get(item_name, 0)
            return details
        return None
        
    def refresh_data(self) -> None:
        """Reload menu and inventory data from files."""
        self._load_data()

# Initialize loader at module level
menu_loader = MenuLoader() 
