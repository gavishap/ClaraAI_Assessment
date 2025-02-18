from typing import Dict, List, Tuple, Optional, Set
import json
from loguru import logger
from openai import OpenAI
import numpy as np
from functools import lru_cache
from datetime import datetime

from ..models import Order, OrderItem
from ..config import OPENAI_CONFIG, MENU_ITEMS
from ..utils.fuzzy_matching import find_best_match, calculate_similarity
from .context_manager import context_manager

class ValidationResult:
    def __init__(self):
        self.is_valid = True
        self.issues: List[str] = []
        self.suggestions: Dict[str, List[str]] = {}
        self.requires_user_input = False
        self.user_queries: List[Dict] = []
        self.validation_steps: List[Dict] = []  # Track each validation step

    def add_validation_step(self, step_type: str, details: Dict):
        """Add a validation step with details."""
        self.validation_steps.append({
            "type": step_type,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })

    def add_issue(self, issue: str):
        self.issues.append(issue)
        self.is_valid = False
        logger.warning(f"Validation issue: {issue}")

    def add_suggestion(self, item: str, suggestions: List[str]):
        self.suggestions[item] = suggestions
        logger.info(f"Added suggestions for '{item}': {suggestions}")

    def add_user_query(self, query_type: str, item: str, options: List[str]):
        self.requires_user_input = True
        self.user_queries.append({
            "type": query_type,
            "item": item,
            "options": options
        })
        logger.info(f"Added user query - Type: {query_type}, Item: {item}, Options: {options}")

class EnhancedValidator:
    def __init__(self, menu_items: Dict):
        logger.info("Initializing EnhancedValidator")
        self.menu_items = menu_items
        self.client = OpenAI(api_key=OPENAI_CONFIG["api_key"])
        
        # Create O(1) lookup dictionaries
        logger.info("Creating lookup dictionaries...")
        self.item_dict = self._create_item_dict()
        self.modification_dict = self._create_modification_dict()
        logger.info(f"Created item dictionary with {len(self.item_dict)} items")
        logger.info(f"Created modification dictionary with {len(self.modification_dict)} modifications")
        
        # Initialize embedding cache
        self.embedding_cache = {}

    def _create_item_dict(self) -> Dict[str, Tuple[str, Dict]]:
        """Create O(1) lookup dictionary for menu items."""
        item_dict = {}
        for category, items in self.menu_items["categories"].items():
            for item_name, details in items.items():
                item_dict[item_name.lower()] = (category, details)
        return item_dict

    def _create_modification_dict(self) -> Dict[str, Set[str]]:
        """Create O(1) lookup dictionary for modifications."""
        mod_dict = {}
        for category, items in self.menu_items["categories"].items():
            for item_name, details in items.items():
                if details["modifications_allowed"]:
                    for mod in details["available_modifications"]:
                        mod_dict[mod.lower()] = item_name
        return mod_dict

    @lru_cache(maxsize=100)
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get OpenAI embedding for text."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None

    def _calculate_embedding_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two text embeddings."""
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        
        if emb1 is None or emb2 is None:
            return 0.0
            
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def validate_item(self, item: OrderItem) -> ValidationResult:
        """Validate a single order item with multiple strategies."""
        result = ValidationResult()
        logger.info(f"\n{'='*50}\nStarting validation for item: {item.name}\n{'='*50}")

        # Step 1: O(1) dictionary lookup
        logger.info("\n1. Performing O(1) dictionary lookup...")
        item_key = item.name.lower()
        if item_key in self.item_dict:
            logger.info(f"✓ Success: Item '{item.name}' found in menu (direct match)")
            result.add_validation_step("dictionary_lookup", {
                "item": item.name,
                "status": "success",
                "match_type": "direct"
            })
            category, details = self.item_dict[item_key]
            return self._validate_modifications(item, details, result)

        logger.info("✗ Item not found in dictionary, proceeding to fuzzy matching")
        result.add_validation_step("dictionary_lookup", {
            "item": item.name,
            "status": "failed"
        })

        # Step 2: Fuzzy matching
        logger.info("\n2. Attempting fuzzy matching...")
        menu_items = list(self.item_dict.keys())
        matched_item, score = find_best_match(item.name, menu_items)
        
        if matched_item and score > 0.8:
            logger.info(f"✓ Success: Item '{item.name}' matched to '{matched_item}' (fuzzy match, score: {score:.2f})")
            result.add_validation_step("fuzzy_matching", {
                "item": item.name,
                "matched_to": matched_item,
                "score": score,
                "status": "success"
            })
            category, details = self.item_dict[matched_item.lower()]
            result.add_suggestion(item.name, [matched_item])
            return self._validate_modifications(item, details, result)

        logger.info("✗ No good fuzzy matches found, proceeding to embedding similarity")
        result.add_validation_step("fuzzy_matching", {
            "item": item.name,
            "status": "failed",
            "best_score": score if matched_item else 0.0
        })

        # Step 3: Embedding similarity
        logger.info("\n3. Performing embedding similarity search...")
        similar_items = []
        for menu_item in menu_items:
            similarity = self._calculate_embedding_similarity(item.name, menu_item)
            if similarity > 0.85:
                similar_items.append((menu_item, similarity))
                logger.info(f"Found similar item: {menu_item} (similarity: {similarity:.2f})")

        if similar_items:
            similar_items.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"✓ Success: Found {len(similar_items)} similar items using embeddings")
            logger.info(f"Top matches: {similar_items[:3]}")
            result.add_validation_step("embedding_similarity", {
                "item": item.name,
                "status": "success",
                "matches": similar_items[:3]
            })
            result.add_suggestion(
                item.name,
                [item[0] for item in similar_items[:3]]
            )
            result.add_user_query(
                "item_replacement",
                item.name,
                [item[0] for item in similar_items[:3]]
            )
        else:
            logger.warning(f"✗ No similar items found for '{item.name}' using any method")
            result.add_validation_step("embedding_similarity", {
                "item": item.name,
                "status": "failed"
            })
            result.add_issue(f"Item '{item.name}' not found in menu")
            result.add_user_query(
                "item_removal",
                item.name,
                []
            )

        return result

    def _validate_modifications(self, item: OrderItem, item_details: Dict, result: ValidationResult) -> ValidationResult:
        """Validate modifications for an item."""
        logger.info(f"\nValidating modifications for {item.name}...")
        
        if not item.modifications:
            logger.info("No modifications to validate")
            return result

        if not item_details["modifications_allowed"]:
            logger.warning(f"❌ Modifications are not allowed for {item.name}")
            result.add_issue(f"Modifications are not allowed for {item.name}")
            return result

        available_mods = set(mod.lower() for mod in item_details["available_modifications"])
        logger.info(f"Available modifications: {available_mods}")
        
        for mod in item.modifications:
            mod_key = mod.lower()
            logger.info(f"\nChecking modification: '{mod}'")
            
            # Step 1: O(1) dictionary lookup
            if mod_key in available_mods:
                logger.info(f"✓ Success: Modification '{mod}' is valid (direct match)")
                result.add_validation_step("modification_lookup", {
                    "modification": mod,
                    "status": "success",
                    "match_type": "direct"
                })
                continue

            # Step 2: Fuzzy matching
            matched_mod, score = find_best_match(mod, list(available_mods))
            if matched_mod and score > 0.8:
                logger.info(f"✓ Success: Modification '{mod}' matched to '{matched_mod}' (fuzzy match, score: {score:.2f})")
                result.add_validation_step("modification_fuzzy", {
                    "modification": mod,
                    "matched_to": matched_mod,
                    "score": score,
                    "status": "success"
                })
                result.add_suggestion(mod, [matched_mod])
                continue

            # Step 3: Ask user
            logger.warning(f"❌ Invalid modification '{mod}' for {item.name}")
            result.add_validation_step("modification_validation", {
                "modification": mod,
                "status": "failed",
                "available_options": list(available_mods)
            })
            result.add_issue(f"Invalid modification '{mod}' for {item.name}")
            result.add_user_query(
                "modification_removal",
                mod,
                list(available_mods)
            )

        return result

    def validate_order(self, order: Order, inventory: Dict[str, int]) -> ValidationResult:
        """Validate entire order."""
        result = ValidationResult()
        logger.info("\n=== Starting Order Validation ===")

        # Validate each item
        for item in order.items:
            item_result = self.validate_item(item)
            result.issues.extend(item_result.issues)
            result.suggestions.update(item_result.suggestions)
            result.user_queries.extend(item_result.user_queries)
            result.requires_user_input |= item_result.requires_user_input
            result.validation_steps.extend(item_result.validation_steps)

        # Validate inventory levels
        logger.info("\n=== Validating Inventory Levels ===")
        self._validate_inventory(order, inventory, result)

        # Log validation summary
        logger.info("\n=== Validation Summary ===")
        logger.info(f"Valid: {result.is_valid}")
        logger.info(f"Issues: {len(result.issues)}")
        logger.info(f"Suggestions: {len(result.suggestions)}")
        logger.info(f"Requires user input: {result.requires_user_input}")

        # Update context with validation results
        context_manager.update_current_context(
            validation_issue={
                "issues": result.issues,
                "suggestions": result.suggestions,
                "requires_user_input": result.requires_user_input,
                "user_queries": result.user_queries,
                "validation_steps": result.validation_steps
            }
        )

        return result

    def _validate_inventory(self, order: Order, inventory: Dict[str, int], result: ValidationResult):
        """Validate inventory levels."""
        for item in order.items:
            if item.name in inventory:
                if inventory[item.name] < item.quantity:
                    result.add_issue(
                        f"Insufficient inventory for {item.name}. "
                        f"Only {inventory[item.name]} available."
                    )
                    result.add_user_query(
                        "quantity_adjustment",
                        item.name,
                        [str(inventory[item.name])]
                    )

# Initialize validator at module level
enhanced_validator = EnhancedValidator(menu_items=MENU_ITEMS) 
