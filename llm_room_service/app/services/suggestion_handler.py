from typing import Dict, List
import json
from openai import OpenAI
from loguru import logger

from ..config import OPENAI_CONFIG
from ..models import Order, OrderItem, OrderIntent
from .enhanced_validation import enhanced_validator
from .langchain_context import langchain_context
from .state_machine import state_machine
from .order_state import OrderState

class SuggestionHandler:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_CONFIG["api_key"])

    async def handle_suggestion_response(self, text: str, context: Dict) -> Dict:
        """Handle user's response to a suggestion."""
        logger.info(f"\nHandling suggestion response: {text}")
        logger.info(f"Context received: {context}")
        
        # Get current order and query from context
        current_order = context.get("current_order", {})
        current_query = context.get("query", {})
        
        logger.info(f"Current order: {current_order}")
        logger.info(f"Current query: {current_query}")
        
        # If no current order, initialize one
        if not current_order:
            current_order = {"items": []}
            logger.info("Initialized new order")
            
        # If no current query, try to get from recent suggestions
        if not current_query:
            recent_suggestions = context.get("recent_suggestions", [])
            logger.info(f"Recent suggestions: {recent_suggestions}")
            
            if recent_suggestions:
                latest_suggestion = None
                for suggestion in reversed(recent_suggestions):
                    if isinstance(suggestion, dict) and "item" in suggestion and "suggestions" in suggestion:
                        latest_suggestion = suggestion
                        break
                        
                if latest_suggestion:
                    # Determine query type based on suggestion context
                    query_type = "modification_replacement" if any(
                        issue.get("message", "").startswith("Modification") 
                        for issue in context.get("recent_issues", [])
                    ) else "item_replacement"
                    
                    current_query = {
                        "type": query_type,
                        "item": latest_suggestion["item"],
                        "suggestions": latest_suggestion["suggestions"],
                        "modifications_required": False
                    }
                    logger.info(f"Reconstructed query from recent suggestion: {current_query}")
                    # Update both contexts with the reconstructed query
                    context["query"] = current_query
                    langchain_context.update_order_memory(query=current_query)
                    state_machine.update_context(type('Event', (), {'kwargs': {'context': {'query': current_query}}})())
                    
        if not current_query:
            logger.error("No active query found")
            return {
                "success": False,
                "error": "No active query found. Please try your request again."
            }
            
        # Check if we're handling an item replacement response
        if current_query.get("type") == "item_replacement":
            logger.info("Processing item replacement response")
            # Find the best match for the user's response
            suggestions = [s[0] for s in current_query["suggestions"]]
            logger.info(f"Available suggestions: {suggestions}")
            
            from ..utils.fuzzy_matching import find_best_match
            best_match, score = find_best_match(text, suggestions)
            logger.info(f"Best match: {best_match}, score: {score}")
            
            if best_match:
                # Update the current order with the selected item
                for item in current_order["items"]:
                    if item["name"] == current_query["item"]:
                        item["name"] = best_match
                        logger.info(f"Updated item in order: {item}")
                        break
                        
                # Create transition context with both order and query
                transition_context = {
                    "order": current_order,
                    "query": current_query  # Keep query for state transition
                }
                
                # Update both contexts with the modified order
                context["current_order"] = current_order
                langchain_context.update_order_memory(current_order=current_order)
                
                # Transition to modification selection state with full context
                state_machine.transition_to(
                    OrderState.MODIFICATION_SELECTION,
                    "Item selected, awaiting modifications",
                    transition_context
                )
                
                # Only clear query after successful transition
                context["query"] = None
                langchain_context.update_order_memory(query=None)
                
                return {
                    "success": True,
                    "message": "Item updated successfully, awaiting modifications.",
                    "order": current_order
                }
            else:
                logger.error("No valid item found for replacement")
                return {
                    "success": False,
                    "error": "No valid item found for replacement."
                }

        # Check if we're handling a modification response
        if current_query.get("type") == "modification_replacement":
            logger.info("Processing modification replacement response")
            
            # Create prompt for GPT to interpret the modification response
            prompt = self._create_modification_prompt(current_query, context)
            
            # Get GPT's interpretation of the modifications
            response = self.client.chat.completions.create(
                model=OPENAI_CONFIG["model"],
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                response_format={ "type": "json_object" },
                temperature=0.3
            )
            
            # Parse the interpretation
            interpretation = json.loads(response.choices[0].message.content)
            logger.info(f"GPT interpretation: {interpretation}")
            
            # Update the order with new modifications
            result = self._process_interpretation(interpretation, current_order, current_query)
            logger.info(f"Processed interpretation result: {result}")
            
            if result["success"]:
                # Update context with the modified order
                langchain_context.update_order_memory(current_order=result["order"])
                
                try:
                    # Transition to modification validation state
                    state_machine.transition_to(
                        OrderState.MODIFICATION_VALIDATION,
                        "Validating modifications",
                        {"order": result["order"], "query": current_query}
                    )
                    return result
                except Exception as e:
                    logger.error(f"Error handling modification response: {str(e)}")
                    return {
                        "success": False,
                        "error": "Failed to process your modifications. Please try again."
                    }
            
            return result
            
        # If not handling modifications, proceed with normal suggestion handling
        logger.info("Processing regular suggestion response")
        
        # Create prompt for GPT
        prompt = self._create_suggestion_prompt(current_query, context)
        
        # Get GPT's interpretation
        response = self.client.chat.completions.create(
            model=OPENAI_CONFIG["model"],
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            response_format={ "type": "json_object" },
            temperature=0.3
        )
        
        # Parse the interpretation
        interpretation = json.loads(response.choices[0].message.content)
        logger.info(f"GPT interpretation: {interpretation}")
        
        # Process the interpretation and update the order
        result = self._process_interpretation(interpretation, current_order, current_query)
        logger.info(f"Processed interpretation result: {result}")
        
        # If successful, update the state machine context
        if result["success"]:
            # Update context with the modified order
            langchain_context.update_order_memory(current_order=result["order"])
            
            try:
                # First validate the selected item
                state_machine.transition_to(
                    OrderState.ITEM_VALIDATION,
                    "Validating selected item",
                    {"order": result["order"], "query": current_query}
                )
                
                # If item is valid, proceed to modification validation
                if result.get("modifications_required"):
                    state_machine.transition_to(
                        OrderState.MODIFICATION_VALIDATION,
                        "Checking for modifications",
                        {"order": result["order"], "query": current_query}
                    )
                
                return result
            except Exception as e:
                logger.error(f"Error handling suggestion response: {str(e)}")
                return {
                    "success": False,
                    "error": "Failed to process your response. Please try again."
                }
        
        return result

    def _create_suggestion_prompt(self, current_query: Dict, context: Dict) -> str:
        """Create a prompt for GPT to interpret the user's response to a suggestion."""
        if current_query["type"] == "item_replacement":
            return self._create_item_replacement_prompt(current_query, context)
        elif current_query["type"] == "modification_replacement":
            return self._create_modification_prompt(current_query, context)
        else:
            return "Unknown query type"
            
    def _create_item_replacement_prompt(self, query: Dict, context_summary: Dict) -> str:
        """Create a prompt for handling item replacement responses."""
        suggestions = query.get("suggestions", [])
        options = "\n".join(f"{i+1}. {suggestion[0]} (score: {suggestion[1]:.2f})" 
                          for i, suggestion in enumerate(suggestions))
                          
        return f"""You are helping to interpret a user's response to item suggestions.

Current context:
- Original item: {query['item']}
- Available options:
{options}

The user can:
1. Select an option by number (e.g., "1" or "first one")
2. Select an option by name (e.g., "garden salad")
3. Remove the item (by saying "remove" or similar)

Respond with a JSON object:
{{
    "action": "select" or "remove",
    "selected_item": "name of selected item" (null if removing),
    "confidence": float between 0 and 1
}}

Only select an item if you're confident it matches one of the available options."""
            
    def _process_interpretation(self, interpretation: Dict, current_order: Dict, current_query: Dict) -> Dict:
        """Process GPT's interpretation of the user's response."""
        logger.info(f"\nProcessing interpretation: {interpretation}")
        logger.info(f"Current query type: {current_query.get('type')}")
        logger.info(f"Current order: {current_order}")
        logger.info(f"Current query: {current_query}")
        
        # Validate the current query
        if not current_query or 'type' not in current_query or 'item' not in current_query:
            return {
                "success": False,
                "error": "Invalid query format"
            }
        
        if current_query['type'] == "modification_replacement":
            # Handle modification replacement
            if interpretation.get('action') == "select":
                selected_mod = interpretation.get('selected_item')
                if not selected_mod:
                    return {
                        "success": False,
                        "error": "No modification selected"
                    }
                    
                # Update the modifications for the item
                for item in current_order["items"]:
                    if item.get("modifications"):
                        # Replace the old modification with the new one
                        old_mod = current_query.get("item")
                        item["modifications"] = [
                            mod if mod != old_mod else selected_mod 
                            for mod in item["modifications"]
                        ]
                
                return {
                    "success": True,
                    "order": current_order,
                    "message": f"Updated modification to {selected_mod}",
                    "modifications_required": False
                }
                
            elif interpretation.get('action') == "remove":
                # Remove the modification from the item
                for item in current_order["items"]:
                    if item.get("modifications"):
                        item["modifications"] = [
                            mod for mod in item["modifications"]
                            if mod != current_query.get("item")
                        ]
                
                return {
                    "success": True,
                    "order": current_order,
                    "message": f"Removed modification: {current_query.get('item')}",
                    "modifications_required": False
                }
            
            return {
                "success": False,
                "error": "Invalid modification action"
            }
            
        elif current_query['type'] == "item_replacement":
            # Handle item replacement/removal
            if interpretation.get('action') == "select":
                selected_item = interpretation.get('selected_item')
                if not selected_item:
                    return {
                        "success": False,
                        "error": "No item selected"
                    }
                
                # Find the matching suggestion to verify the selection
                matching_suggestion = None
                for name, score in current_query.get('suggestions', []):
                    if name.lower() == selected_item.lower():
                        matching_suggestion = name
                        break
                
                if not matching_suggestion:
                    return {
                        "success": False,
                        "error": f"Selected item '{selected_item}' not found in suggestions"
                    }
                
                # Update the item name while preserving modifications
                for item in current_order["items"]:
                    if item["name"] == current_query.get("item"):
                        item["name"] = matching_suggestion
                        
                return {
                    "success": True,
                    "order": current_order,
                    "message": f"Updated your order with {matching_suggestion}",
                    "modifications_required": bool(current_order["items"][0].get("modifications", []))
                }
                
            elif interpretation.get('action') == "remove":
                # Remove the item from the order
                current_order["items"] = [
                    item for item in current_order["items"]
                    if item["name"] != current_query.get("item")
                ]
                
                return {
                    "success": True,
                    "order": current_order,
                    "message": f"Removed {current_query.get('item')} from your order",
                    "modifications_required": False
                }
        
        return {
            "success": False,
            "error": f"Unsupported query type: {current_query.get('type')}"
        }

    def _create_modification_prompt(self, query: Dict, context_summary: Dict) -> str:
        """Create a prompt for modification-related queries."""
        suggestions = query.get('suggestions', [])
        options = "\n".join(f"{i+1}. {suggestion[0]} (score: {suggestion[1]:.2f})" 
                          for i, suggestion in enumerate(suggestions))
        
        current_order = context_summary.get("current_order", {})
        current_items = [
            f"- {item['name']} with modifications: {', '.join(item['modifications'])}"
            for item in current_order.get("items", [])
            if item.get("modifications")
        ]
        current_order_str = "\n".join(current_items) if current_items else "No items with modifications"
        
        return f"""You are helping to interpret a user's response to modification suggestions.

Current context:
- Order details:
{current_order_str}
- Modification to replace: {query['item']}
- Available options:
{options}

The user can:
1. Select an option by number (e.g., "1" or "first one")
2. Select an option by name (e.g., "avocado")
3. Remove the modification (by saying "remove" or similar)

Respond with a JSON object:
{{
    "action": "select" or "remove",
    "selected_item": "name of selected modification" (null if removing),
    "confidence": float between 0 and 1
}}

Only select a modification if you're confident it matches one of the available options.
If the user's input exactly matches or is very similar to one of the options, select that option."""

    def _create_order_from_dict(self, order_dict: Dict) -> Order:
        """Create an Order object from a dictionary representation."""
        items = []
        for item_dict in order_dict["items"]:
            items.append(OrderItem(
                name=item_dict["name"],
                quantity=item_dict["quantity"],
                modifications=item_dict["modifications"],
                category=item_dict.get("category", "Main")  # Default to Main if not specified
            ))
            
        return Order(
            items=items,
            intent=OrderIntent.NEW_ORDER,
            room_number=order_dict.get("room_number")
        )

    def _handle_validation_result(self, validation_result: Dict, updated_order: Dict, 
                                pending_queries: List[Dict], order: Order, inventory: Dict) -> Dict:
        """Handle the validation result after processing a suggestion."""
        if validation_result["is_valid"] and not validation_result["requires_user_input"]:
            if not pending_queries:  # No more queries to handle
                return self._finalize_order(order, updated_order, inventory)
            else:
                # Format the next query
                return self._format_validation_prompts(validation_result, updated_order)
        else:
            # There are still validation issues
            state_machine.transition_to(
                OrderState.ITEM_VALIDATION,
                "Validation issues found",
                {"order": updated_order}
            )
            return self._format_validation_prompts(validation_result, updated_order)

    def _format_validation_prompts(self, validation_result: Dict, updated_order: Dict) -> Dict:
        """Format validation prompts for user interaction."""
        prompts = []
        for query in validation_result.get("user_queries", []):
            if query["type"] == "item_replacement":
                options = "\n".join(f"{i+1}. {name}" 
                                  for i, (name, _) in enumerate(query["suggestions"]))
                prompts.append(
                    f"For '{query['item']}', please choose one of these options "
                    f"or type 'remove' to remove it:\n{options}"
                )
            elif query["type"] == "modification_replacement":
                options = "\n".join(f"- {name}" for name, _ in query["suggestions"])
                prompts.append(
                    f"Available modifications for {query['item']}:\n{options}\n\n"
                    "What modifications would you like? (You can choose multiple)"
                )
                
        return {
            "success": False,
            "type": "validation",
            "order": updated_order,
            "prompts": prompts
        }

    def _finalize_order(self, order: Order, updated_order: Dict, inventory: Dict) -> Dict:
        """Finalize the order if all validations pass."""
        # Perform final validation
        validation_result = enhanced_validator.validate_order(order, inventory)
        
        if validation_result["is_valid"]:
            state_machine.transition_to(
                OrderState.ORDER_CONFIRMATION,
                "Order validated",
                {"order": updated_order}
            )
            return {
                "success": True,
                "type": "order",
                "order": updated_order,
                "message": "Order updated successfully!"
            }
        else:
            state_machine.transition_to(
                OrderState.ERROR,
                "Final validation failed",
                {"error": validation_result["issues"][0]}
            )
            return {
                "success": False,
                "error": "Final validation failed",
                "issues": validation_result["issues"]
            }

    def _remove_item_from_order(self, order: Dict, item_name: str) -> Dict:
        """Remove an item from the order."""
        order["items"] = [item for item in order["items"] if item["name"] != item_name]
        return order

    def _replace_item_in_order(self, order: Dict, old_item: str, new_item: str) -> Dict:
        """Replace an item in the order with a new item."""
        for item in order["items"]:
            if item["name"] == old_item:
                item["name"] = new_item
                item["modifications"] = []  # Clear modifications as they might not apply
        return order

# Initialize handler at module level
suggestion_handler = SuggestionHandler() 
