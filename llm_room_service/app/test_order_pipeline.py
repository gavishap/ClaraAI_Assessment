import asyncio
import json
from typing import Dict
from loguru import logger
from openai import OpenAI

from .services.order_extraction import OrderExtractor
from .services.enhanced_validation import enhanced_validator
from .services.intent_classifier import intent_classifier
from .services.menu_inquiry import menu_inquiry_system
from .services.suggestion_handler import suggestion_handler
from .services.state_machine import state_machine
from .services.order_state import OrderState
from .services.langchain_context import langchain_context
from .models import Order, OrderIntent, OrderItem
from .config import MENU_ITEMS, OPENAI_CONFIG
from .utils.response_formatter import response_formatter

async def process_order_text(text: str) -> Dict:
    """Process natural language order text through extraction and validation."""
    # Add user message to context
    langchain_context.add_user_message(text)
    
    # Get current state
    current_state = state_machine.get_current_state()
    logger.info(f"Current state: {current_state}")
    
    # Get current context
    context = langchain_context.get_order_context()
    logger.info(f"Current context: {context}")
    
    # If we're in modification selection state, handle it directly without intent classification
    if current_state == OrderState.MODIFICATION_SELECTION:
        logger.info("Processing modification selection...")
        # Get current context
        context = langchain_context.get_order_context()
        logger.info(f"Current context: {context}")
        
        if not context or not context.get("current_order"):
            logger.error("No active order found in context")
            return {
                "success": False,
                "error": "No active order found"
            }
            
        # Get the current query from state machine context
        state_context = state_machine.get_context()
        logger.info(f"State machine context: {state_context}")
        
        # Ensure we have a valid query to process
        current_query = None
        if state_context and "query" in state_context:
            current_query = state_context["query"]
            logger.info(f"Found query in state context: {current_query}")
        else:
            # Try to get the most recent suggestion from context
            recent_suggestions = context.get("recent_suggestions", [])
            logger.info(f"Recent suggestions: {recent_suggestions}")
            
            for suggestion in reversed(recent_suggestions):
                if "item" in suggestion and "suggestions" in suggestion:
                    current_query = {
                        "type": "modification_replacement",
                        "item": suggestion["item"],
                        "suggestions": suggestion["suggestions"]
                    }
                    logger.info(f"Created query from recent suggestion: {current_query}")
                    break
                    
        if not current_query:
            logger.error("No valid query found for modification selection")
            return {
                "success": False,
                "error": "No pending modification selection found"
            }
            
        # Add the current query to the context for the suggestion handler
        context["current_query"] = current_query
        logger.info(f"Updated context with query: {context['current_query']}")
        
        # Use suggestion handler to process the response
        logger.info(f"Calling suggestion handler with text: {text}")
        result = await suggestion_handler.handle_suggestion_response(text, context)
        logger.info(f"Suggestion handler result: {result}")
        
        if result["success"]:
            # Update context with the modified order
            langchain_context.update_order_memory(current_order=result["order"])
            
            # Create order object for validation
            order_obj = Order(
                items=[
                    OrderItem(**item)
                    for item in result["order"]["items"]
                ],
                intent=OrderIntent.NEW_ORDER,
                room_number=result["order"].get("room_number")
            )
            
            # Load inventory for validation
            with open("llm_room_service/data/inventory.json", "r") as f:
                inventory = json.load(f)
                
            # Validate the updated order
            validation_result = enhanced_validator.validate_order(order_obj, inventory)
            
            if validation_result.is_valid and not validation_result.requires_user_input:
                # Transition to quantity validation if no more modifications needed
                state_machine.transition_to(
                    OrderState.QUANTITY_VALIDATION,
                    "Validating quantity",
                    {"order": result["order"]}
                )
                return {
                    "success": True,
                    "type": "order",
                    "order": result["order"],
                    "message": "Modifications updated successfully!"
                }
            else:
                # If there are still validation issues, stay in modification selection
                state_machine.transition_to(
                    OrderState.MODIFICATION_SELECTION,
                    "Awaiting modification selection",
                    {"order": result["order"], "query": current_query}
                )
                return {
                    "success": False,
                    "type": "order",
                    "order": result["order"],
                    "validation": {
                        "passed": False,
                        "requires_user_input": True,
                        "prompts": validation_result.user_queries
                    }
                }
        
        return result
    
    # If we're in item selection state, handle the selection
    if current_state == OrderState.ITEM_SELECTION:
        logger.info("Processing item selection...")
        
        # Get current context
        context = langchain_context.get_order_context()
        logger.info(f"Current context: {context}")
        
        if not context or not context.get("current_order"):
            logger.error("No active order found in context")
            return {
                "success": False,
                "error": "No active order found"
            }
            
        # Get the current query from state machine context
        state_context = state_machine.get_context()
        logger.info(f"State machine context: {state_context}")
        
        # Ensure we have a valid query to process
        current_query = None
        if state_context and "query" in state_context:
            current_query = state_context["query"]
            logger.info(f"Found query in state context: {current_query}")
        else:
            # Try to get the most recent suggestion from context
            recent_suggestions = context.get("recent_suggestions", [])
            logger.info(f"Recent suggestions: {recent_suggestions}")
            
            if recent_suggestions:
                latest_suggestion = None
                for suggestion in reversed(recent_suggestions):
                    if isinstance(suggestion, dict) and "item" in suggestion and "suggestions" in suggestion:
                        latest_suggestion = suggestion
                        break
                
                if latest_suggestion:
                    current_query = {
                        "type": "item_replacement",
                        "item": latest_suggestion["item"],
                        "suggestions": latest_suggestion["suggestions"]
                    }
                    logger.info(f"Created query from recent suggestion: {current_query}")
                    # Update both contexts with the reconstructed query
                    state_machine.update_context(type('Event', (), {'kwargs': {'context': {'query': current_query}}})())
                    langchain_context.update_order_memory(query=current_query)
                    
        if not current_query:
            logger.error("No valid query found for item selection")
            return {
                "success": False,
                "error": "No pending item selection found"
            }
            
        # Add the current query to the context for the suggestion handler
        context["query"] = current_query
        logger.info(f"Added query to context: {current_query}")
        
        # Process the selection
        result = await suggestion_handler.handle_suggestion_response(text, context)
        logger.info(f"Suggestion handler result: {result}")
        
        if result["success"]:
            # Update context with the modified order
            langchain_context.update_order_memory(current_order=result["order"])
            
            # Create order object for validation
            order_obj = Order(
                items=[
                    OrderItem(**item)
                    for item in result["order"]["items"]
                ],
                intent=OrderIntent.NEW_ORDER,
                room_number=result["order"].get("room_number")
            )
            
            # Load inventory for validation
            with open("llm_room_service/data/inventory.json", "r") as f:
                inventory = json.load(f)
                
            # Validate the updated order
            validation_result = enhanced_validator.validate_order(order_obj, inventory)
            
            if validation_result.is_valid and not validation_result.requires_user_input:
                state_machine.transition_to(
                    OrderState.ORDER_COMPLETED,
                    "Order completed",
                    {"order": result["order"]}
                )
                return {
                    "success": True,
                    "type": "order",
                    "order": result["order"],
                    "message": result.get("message", "Order updated successfully")
                }
            else:
                state_machine.transition_to(
                    OrderState.ITEM_VALIDATION,
                    "Validating updated order",
                    {"order": result["order"]}
                )
                return {
                    "success": False,
                    "type": "validation",
                    "order": result["order"],
                    "validation": {
                        "passed": False,
                        "requires_user_input": True,
                        "prompts": validation_result.user_queries
                    }
                }
        else:
            return result
    
    # Start in INITIAL state if not already in a state
    if current_state == OrderState.INITIAL:
        state_machine.transition_to(
            OrderState.INTENT_CLASSIFICATION,
            "Starting intent classification",
            {"text": text}
        )
        langchain_context.start_new_conversation()
    
    # Classify intent
    intent, confidence = intent_classifier.classify(text)
    logger.info(f"Classified intent: {intent} (confidence: {confidence:.2f})")
    
    # Handle different intents
    if intent == OrderIntent.UNSUPPORTED_ACTION:
        state_machine.transition_to(
            OrderState.ERROR,
            "Unsupported action",
            {"error": "This type of request is not supported"}
        )
        return {
            "success": False,
            "error": "This type of request is not supported. I can help you place new orders or answer questions about our menu."
        }
    elif intent == OrderIntent.UNKNOWN:
        state_machine.transition_to(
            OrderState.ERROR,
            "Unknown intent",
            {"error": "Unclear request"}
        )
        return {
            "success": False,
            "error": "I'm not sure what you're asking for. Could you please clarify your request? You can place an order or ask about our menu."
        }
    elif intent == OrderIntent.GENERAL_INQUIRY:
        state_machine.transition_to(
            OrderState.MENU_INQUIRY,
            "Handling menu inquiry",
            {"text": text}
        )
        langchain_context.set_state_prompt("menu_inquiry")
        
        answer = await menu_inquiry_system.answer_inquiry(text)
        langchain_context.add_assistant_message(answer)
        
        state_machine.transition_to(
            OrderState.INITIAL,
            "Completed menu inquiry",
            {"response": answer}
        )
        return {
            "success": True,
            "type": "inquiry",
            "response": answer
        }

    # Handle new order intent
    if intent == OrderIntent.NEW_ORDER:
        state_machine.transition_to(
            OrderState.ORDER_EXTRACTION,
            "Extracting order",
            {"text": text}
        )
        langchain_context.set_state_prompt("order_extraction")
        
        # Extract order
        extractor = OrderExtractor()
        order = extractor.extract_order(text, MENU_ITEMS)
        
        if not order:
            state_machine.transition_to(
                OrderState.ERROR,
                "Order extraction failed",
                {"error": "Failed to extract order"}
            )
            logger.error("Failed to extract order")
            return {
                "success": False,
                "error": "Failed to extract order from text"
            }

        logger.info("✓ Order extracted successfully")
        logger.info(f"Extracted order: {order.model_dump_json(indent=2)}")
        
        # Start tracking order in context
        langchain_context.start_new_order(text)
        langchain_context.update_order_memory(
            current_order=order.model_dump()
        )
        
        # Load inventory
        with open("llm_room_service/data/inventory.json", "r") as f:
            inventory = json.load(f)
        
        # Move to validation state
        state_machine.transition_to(
            OrderState.ITEM_VALIDATION,
            "Validating order",
            {"order": order.model_dump()}
        )
        langchain_context.set_state_prompt("validation")
        
        # Validate order
        validation_result = enhanced_validator.validate_order(order, inventory)
        
        # Handle validation result
        if validation_result.is_valid and not validation_result.requires_user_input:
            logger.info("✓ Order validation passed")
            
            # Update context and transition to completed state
            langchain_context.update_order_memory(
                validation_issue={"message": "Order validation passed"}
            )
            state_machine.transition_to(
                OrderState.ORDER_COMPLETED,
                "Order completed",
                {"order": order.model_dump()}
            )
            
            # Clear context since order is complete
            langchain_context.clear_order_memory()
            
            return {
                "success": True,
                "type": "order",
                "order": order.model_dump(),
                "validation": {
                    "passed": True,
                    "suggestions": validation_result.suggestions
                }
            }
        else:
            logger.info("Order requires user input for suggestions")
            
            # Update context with validation results
            for issue in validation_result.issues:
                langchain_context.update_order_memory(
                    validation_issue={"message": issue}
                )
            
            # Add suggestions to context
            for suggestion in validation_result.suggestions:
                langchain_context.update_order_memory(
                    suggestion={"text": suggestion}
                )
            
            # Format user prompts
            prompts = []
            for query in validation_result.user_queries:
                if query["type"] == "item_replacement":
                    if len(query["suggestions"]) > 1:
                        options = "\n".join(f"{i+1}. {name} (score: {score:.2f})" 
                                          for i, (name, score) in enumerate(query["suggestions"]))
                        prompts.append(f"For '{query['item']}', please choose one of these options or type 'remove' to remove it:\n{options}")
                        state_machine.transition_to(
                            OrderState.ITEM_SELECTION,
                            "Awaiting item selection",
                            {"query": query}
                        )
                    else:
                        name, score = query["suggestions"][0]
                        prompts.append(f"Did you mean '{name}' for '{query['item']}'? (yes/no/remove)")
                        state_machine.transition_to(
                            OrderState.ITEM_SELECTION,
                            "Awaiting item confirmation",
                            {"query": query}
                        )
                elif query["type"] == "modification_replacement":
                    options = "\n".join(f"- {name}" for name, _ in query["suggestions"])
                    prompts.append(f"Available modifications:\n{options}\n\nWhat modifications would you like? (You can choose multiple)")
                    state_machine.transition_to(
                        OrderState.MODIFICATION_SELECTION,
                        "Awaiting modification selection",
                        {"query": query}
                    )

            return {
                "success": False,
                "type": "order",
                "order": order.model_dump(),
                "validation": {
                    "passed": False,
                    "requires_user_input": True,
                    "prompts": prompts
                }
            }

async def test_order_pipeline():
    """Interactive test for order processing pipeline."""
    print("\n=== Room Service Order Processing Test ===")
    print("You can:")
    print("1. Place an order (e.g., 'I'd like a Caesar salad with chicken')")
    print("2. Ask about the menu (e.g., 'What's in the Caesar salad?')")
    print("3. Type 'quit' to exit")
    print("\nNote: Other actions like order cancellation are not supported.")
    
    while True:
        text = input("\nYour request: ").strip()
        if text.lower() == 'quit':
            break
            
        print("\nProcessing request...")
        result = await process_order_text(text)
        
        # Format response using the response formatter
        formatted_response = response_formatter.format_response(result)
        print(formatted_response)
        
        # Add assistant response to context
        langchain_context.add_assistant_message(formatted_response)
        
        # Log current state and context
        logger.info(f"Current State: {state_machine.get_current_state()}")
        logger.info(f"Next Expected States: {state_machine.get_next_expected_states()}")
        logger.info("Current Context:")
        logger.info(langchain_context.get_formatted_context())

if __name__ == "__main__":
    asyncio.run(test_order_pipeline()) 
