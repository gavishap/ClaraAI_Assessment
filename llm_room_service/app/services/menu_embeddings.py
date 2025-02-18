from typing import Dict, List, Tuple, Optional
import numpy as np
from openai import OpenAI
from loguru import logger
from functools import lru_cache

from ..config import MENU_ITEMS, OPENAI_CONFIG

class MenuEmbeddingService:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_CONFIG["api_key"])
        self.menu_embeddings = {}
        self.modification_embeddings = {}
        self._initialize_embeddings()
        
    def _initialize_embeddings(self):
        """Initialize embeddings for menu items and modifications."""
        logger.info("Initializing menu embeddings...")
        
        # Create embeddings for menu items
        for category, items in MENU_ITEMS["categories"].items():
            for item_name, details in items.items():
                # Create detailed context for each item
                item_context = f"""
                Item: {item_name}
                Category: {category}
                Description: {details['description']}
                """
                
                try:
                    response = self.client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=item_context.strip()
                    )
                    self.menu_embeddings[item_name] = {
                        "embedding": np.array(response.data[0].embedding),
                        "context": item_context,
                        "category": category,
                        "details": details
                    }
                    
                    # Create embeddings for each modification
                    if details["modifications_allowed"]:
                        for mod in details["available_modifications"]:
                            if mod not in self.modification_embeddings:
                                mod_response = self.client.embeddings.create(
                                    model="text-embedding-ada-002",
                                    input=mod
                                )
                                self.modification_embeddings[mod] = {
                                    "embedding": np.array(mod_response.data[0].embedding),
                                    "items": set()
                                }
                            self.modification_embeddings[mod]["items"].add(item_name)
                            
                except Exception as e:
                    logger.error(f"Error creating embedding for {item_name}: {e}")
        
        logger.info(f"Created embeddings for {len(self.menu_embeddings)} menu items and {len(self.modification_embeddings)} modifications")

    @lru_cache(maxsize=100)
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text string."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None

    def find_similar_items(self, query: str, threshold: float = 0.7) -> List[Tuple[str, float, Dict]]:
        """Find menu items similar to the query."""
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            return []
            
        similar_items = []
        for item_name, data in self.menu_embeddings.items():
            similarity = self._calculate_similarity(query_embedding, data["embedding"])
            if similarity > threshold:
                similar_items.append((item_name, similarity, data))
                
        similar_items.sort(key=lambda x: x[1], reverse=True)
        return similar_items

    def find_similar_modifications(self, query: str, item_name: Optional[str] = None, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find modifications similar to the query, optionally filtered by item."""
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            return []
            
        similar_mods = []
        for mod_name, data in self.modification_embeddings.items():
            # If item_name is provided, only consider modifications available for that item
            if item_name and item_name not in data["items"]:
                continue
                
            similarity = self._calculate_similarity(query_embedding, data["embedding"])
            if similarity > threshold:
                similar_mods.append((mod_name, similarity))
                
        similar_mods.sort(key=lambda x: x[1], reverse=True)
        return similar_mods

    def get_item_details(self, item_name: str) -> Optional[Dict]:
        """Get details for a menu item."""
        if item_name in self.menu_embeddings:
            return self.menu_embeddings[item_name]
        return None

    def get_available_modifications(self, item_name: str) -> List[str]:
        """Get available modifications for an item."""
        if item_name in self.menu_embeddings:
            return [
                mod for mod, data in self.modification_embeddings.items()
                if item_name in data["items"]
            ]
        return []

    @staticmethod
    def _calculate_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Initialize service at module level
menu_embedding_service = MenuEmbeddingService() 
