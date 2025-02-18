from typing import Dict, Any, Optional, List
from loguru import logger
from transitions import Machine
from datetime import datetime
from .order_state import OrderState
from .langchain_context import langchain_context

class OrderStateMachine:
    def __init__(self):
        # Define states
        states = [state.value for state in OrderState]
        
        # Initialize the machine
        self.machine = Machine(
            model=self,
            states=states,
            initial=OrderState.INITIAL.value,
            send_event=True  # Enable event sending
        )
        
        # Add transitions
        self.machine.add_transition(
            trigger='classify_intent',
            source=OrderState.INITIAL.value,
            dest=OrderState.INTENT_CLASSIFICATION.value,
            before='log_transition',
            after='update_context'
        )
        
        self.machine.add_transition(
            trigger='extract_order',
            source=[OrderState.INTENT_CLASSIFICATION.value, OrderState.ERROR.value],
            dest=OrderState.ORDER_EXTRACTION.value,
            before='log_transition',
            after='update_context'
        )
        
        self.machine.add_transition(
            trigger='validate_items',
            source=[
                OrderState.ORDER_EXTRACTION.value,
                OrderState.ITEM_SELECTION.value,
                OrderState.MODIFICATION_SELECTION.value,
                OrderState.ERROR.value
            ],
            dest=OrderState.ITEM_VALIDATION.value,
            before='log_transition',
            after='update_context'
        )
        
        self.machine.add_transition(
            trigger='select_item',
            source=[
                OrderState.ITEM_VALIDATION.value,
                OrderState.ERROR.value
            ],
            dest=OrderState.ITEM_SELECTION.value,
            before='log_transition',
            after='update_context'
        )
        
        self.machine.add_transition(
            trigger='validate_modifications',
            source=[
                OrderState.ITEM_VALIDATION.value,
                OrderState.MODIFICATION_SELECTION.value,
                OrderState.ERROR.value
            ],
            dest=OrderState.MODIFICATION_VALIDATION.value,
            before='log_transition',
            after='update_context'
        )
        
        self.machine.add_transition(
            trigger='select_modifications',
            source=[
                OrderState.MODIFICATION_VALIDATION.value,
                OrderState.MODIFICATION_SELECTION.value,
                OrderState.ERROR.value
            ],
            dest=OrderState.MODIFICATION_SELECTION.value,
            before='log_transition',
            after='update_context'
        )
        
        self.machine.add_transition(
            trigger='validate_quantity',
            source=[
                OrderState.MODIFICATION_VALIDATION.value,
                OrderState.QUANTITY_ADJUSTMENT.value,
                OrderState.ERROR.value
            ],
            dest=OrderState.QUANTITY_VALIDATION.value,
            before='log_transition',
            after='update_context'
        )
        
        self.machine.add_transition(
            trigger='adjust_quantity',
            source=[
                OrderState.QUANTITY_VALIDATION.value,
                OrderState.ERROR.value
            ],
            dest=OrderState.QUANTITY_ADJUSTMENT.value,
            before='log_transition',
            after='update_context'
        )
        
        self.machine.add_transition(
            trigger='confirm_order',
            source=[
                OrderState.QUANTITY_VALIDATION.value,
                OrderState.ERROR.value
            ],
            dest=OrderState.ORDER_CONFIRMATION.value,
            before='log_transition',
            after='update_context'
        )
        
        self.machine.add_transition(
            trigger='complete_order',
            source=[
                OrderState.ORDER_CONFIRMATION.value,
                OrderState.ERROR.value
            ],
            dest=OrderState.ORDER_COMPLETED.value,
            before='log_transition',
            after='update_context'
        )
        
        self.machine.add_transition(
            trigger='handle_error',
            source='*',
            dest=OrderState.ERROR.value,
            before='log_transition',
            after='update_context'
        )
        
        self.machine.add_transition(
            trigger='reset',
            source='*',
            dest=OrderState.INITIAL.value,
            before='log_transition',
            after='clear_context'
        )
        
        # Initialize context
        self._context = {}
        
    def start_new_order(self, text: str) -> None:
        """Start a new order process."""
        langchain_context.start_new_order(text)
        self.classify_intent(context={'text': text})
        
    def update_context(self, event) -> None:
        """Update the context with new data."""
        # Get context from event data
        data = event.kwargs.get('context', {}) if event.kwargs else {}
            
        if not data:
            return
            
        # Initialize context if needed
        if not hasattr(self, '_context'):
            self._context = {}
            
        # Get the current state
        current_state = self.get_current_state()
        logger.info(f"Updating context in state: {current_state}")
        logger.info(f"Current context before update: {self._context}")
        logger.info(f"New data to update: {data}")
        
        # Special handling for ITEM_SELECTION and MODIFICATION_SELECTION states
        if current_state in [OrderState.ITEM_SELECTION, OrderState.MODIFICATION_SELECTION]:
            # If we don't have a query in the new data, try to reconstruct it
            if 'query' not in data or not data['query']:
                # Try to get query from existing context
                if 'query' in self._context:
                    data['query'] = self._context['query']
                    logger.info(f"Retrieved query from existing context: {data['query']}")
                # If not in context, try to reconstruct from recent suggestions
                elif 'recent_suggestions' in self._context:
                    recent_suggestions = self._context['recent_suggestions']
                    logger.info(f"Attempting to reconstruct query from recent suggestions: {recent_suggestions}")
                    if recent_suggestions and isinstance(recent_suggestions, list):
                        latest_suggestion = None
                        for suggestion in reversed(recent_suggestions):
                            if isinstance(suggestion, dict) and "item" in suggestion and "suggestions" in suggestion:
                                latest_suggestion = suggestion
                                break
                        if latest_suggestion:
                            query_type = 'modification_replacement' if current_state == OrderState.MODIFICATION_SELECTION else 'item_replacement'
                            data['query'] = {
                                "type": query_type,
                                "item": latest_suggestion["item"],
                                "suggestions": latest_suggestion["suggestions"]
                            }
                            logger.info(f"Reconstructed query from suggestions: {data['query']}")
                            # Update langchain context with the reconstructed query
                            langchain_context.update_order_memory(query=data['query'])

            # If we have a query, ensure it's properly formatted
            if 'query' in data and isinstance(data['query'], dict):
                if 'item' in data['query'] and 'suggestions' in data['query']:
                    if 'type' not in data['query']:
                        data['query']['type'] = 'item_replacement' if current_state == OrderState.ITEM_SELECTION else 'modification_replacement'
                    logger.info(f"Updated query type to: {data['query']['type']}")
                    # Update langchain context with the updated query
                    langchain_context.update_order_memory(query=data['query'])

        # Preserve existing context that's not being updated
        merged_context = self._context.copy()
        merged_context.update(data)
        self._context = merged_context
        logger.info(f"Updated context: {self._context}")
            
        # Update langchain context
        if 'order' in data:
            langchain_context.update_order_memory(current_order=data['order'])
        if 'issues' in data:
            for issue in data['issues']:
                langchain_context.update_order_memory(validation_issue={"message": issue})
        if 'suggestions' in data:
            for suggestion in data['suggestions']:
                langchain_context.update_order_memory(suggestion={"text": suggestion})
        if 'query' in data and isinstance(data['query'], dict):
            # Update the query in langchain context
            langchain_context.update_order_memory(query=data['query'])
            logger.info(f"Updated query in langchain context: {data['query']}")

    def clear_context(self, event) -> None:
        """Clear the current context."""
        langchain_context.clear_order_memory()
        self._context = {}
        
    def log_transition(self, event) -> None:
        """Log state transitions."""
        # Extract transition details from event
        source_state = event.transition.source
        dest_state = event.transition.dest
        trigger = event.event.name
        
        logger.info(
            f"State transition: {source_state} -> {dest_state} "
            f"(trigger: {trigger})"
        )
        
    # State entry callbacks
    def on_enter_intent_classification(self, event) -> None:
        """Called when entering intent classification state."""
        logger.info("Starting intent classification")
        langchain_context.set_state_prompt("intent_classification")
        
    def on_enter_menu_inquiry(self, event) -> None:
        """Called when entering menu inquiry state."""
        logger.info("Processing menu inquiry")
        langchain_context.set_state_prompt("menu_inquiry")
        
    def on_enter_order_extraction(self, event) -> None:
        """Called when entering order extraction state."""
        logger.info("Starting order extraction")
        langchain_context.set_state_prompt("order_extraction")
        
    def on_enter_error(self, event) -> None:
        """Called when entering error state."""
        logger.error(f"Entered error state. Context: {langchain_context.get_order_context()}")
        
    def get_current_state(self) -> OrderState:
        """Get the current state."""
        return OrderState(self.state)
        
    def get_context(self) -> Optional[Dict]:
        """Get the current context."""
        return langchain_context.get_order_context()
        
    def can_transition_to(self, state: OrderState) -> bool:
        """Check if a transition to the given state is possible."""
        return self.machine.get_triggers(self.state, state.value) != []
        
    def get_next_expected_states(self) -> List[OrderState]:
        """Get list of possible next states from current state."""
        transitions = self.machine.get_transitions(self.state)
        return [OrderState(t.dest) for t in transitions]

    def _get_trigger_for_transition(self, current_state: OrderState, target_state: OrderState) -> Optional[str]:
        """Get the appropriate trigger for transitioning between states."""
        # Define state-to-trigger mapping for common transitions
        state_triggers = {
            (OrderState.INITIAL, OrderState.INTENT_CLASSIFICATION): 'classify_intent',
            (OrderState.INTENT_CLASSIFICATION, OrderState.ORDER_EXTRACTION): 'extract_order',
            (OrderState.ORDER_EXTRACTION, OrderState.ITEM_VALIDATION): 'validate_items',
            (OrderState.ITEM_VALIDATION, OrderState.ITEM_SELECTION): 'select_item',
            (OrderState.ITEM_SELECTION, OrderState.ITEM_VALIDATION): 'validate_items',
            (OrderState.ITEM_VALIDATION, OrderState.MODIFICATION_VALIDATION): 'validate_modifications',
            (OrderState.MODIFICATION_VALIDATION, OrderState.MODIFICATION_SELECTION): 'select_modifications',
            (OrderState.MODIFICATION_SELECTION, OrderState.MODIFICATION_VALIDATION): 'validate_modifications',
            (OrderState.MODIFICATION_VALIDATION, OrderState.QUANTITY_VALIDATION): 'validate_quantity',
            (OrderState.QUANTITY_VALIDATION, OrderState.QUANTITY_ADJUSTMENT): 'adjust_quantity',
            (OrderState.QUANTITY_VALIDATION, OrderState.ORDER_CONFIRMATION): 'confirm_order',
            (OrderState.ORDER_CONFIRMATION, OrderState.ORDER_COMPLETED): 'complete_order',
            # Add error state transitions
            (OrderState.ERROR, OrderState.ORDER_EXTRACTION): 'extract_order',
            (OrderState.ERROR, OrderState.ITEM_VALIDATION): 'validate_items',
            (OrderState.ERROR, OrderState.ITEM_SELECTION): 'select_item',
            (OrderState.ERROR, OrderState.MODIFICATION_VALIDATION): 'validate_modifications',
            (OrderState.ERROR, OrderState.MODIFICATION_SELECTION): 'select_modifications',
            (OrderState.ERROR, OrderState.QUANTITY_VALIDATION): 'validate_quantity',
            (OrderState.ERROR, OrderState.ORDER_CONFIRMATION): 'confirm_order',
            (OrderState.ERROR, OrderState.ORDER_COMPLETED): 'complete_order',
            (OrderState.ERROR, OrderState.INITIAL): 'reset'
        }
        
        # Special case: If we're in ERROR state, use the target state's trigger
        if current_state == OrderState.ERROR:
            error_state_triggers = {
                OrderState.ITEM_VALIDATION: 'validate_items',
                OrderState.MODIFICATION_VALIDATION: 'validate_modifications',
                OrderState.MODIFICATION_SELECTION: 'select_modifications',
                OrderState.QUANTITY_VALIDATION: 'validate_quantity',
                OrderState.ORDER_CONFIRMATION: 'confirm_order',
                OrderState.ORDER_COMPLETED: 'complete_order',
                OrderState.INITIAL: 'reset'
            }
            return error_state_triggers.get(target_state)
            
        # Check direct transitions first
        trigger = state_triggers.get((current_state, target_state))
        if trigger:
            return trigger
            
        # Handle error state transition
        if target_state == OrderState.ERROR:
            return 'handle_error'
            
        # Handle reset to initial state
        if target_state == OrderState.INITIAL:
            return 'reset'
            
        return None

    def transition_to(self, new_state: OrderState, reason: str, context: Optional[Dict] = None) -> None:
        """Transition to a new state with context."""
        try:
            logger.info(f"\nAttempting transition from {self.state} to {new_state.value}")
            logger.info(f"Transition reason: {reason}")
            logger.info(f"New context: {context}")
            
            # Get existing context
            existing_context = self._context.copy()
            logger.info(f"Existing context: {existing_context}")
            
            # Update context
            if context:
                # Merge nested dictionaries instead of overwriting
                for key, value in context.items():
                    if key in existing_context and isinstance(existing_context[key], dict) and isinstance(value, dict):
                        existing_context[key].update(value)
                    else:
                        existing_context[key] = value
            logger.info(f"Updated full context: {existing_context}")
            
            # Find the appropriate trigger
            trigger = self._get_trigger_for_transition(OrderState(self.state), new_state)
            if not trigger:
                raise ValueError(f"No valid trigger found for transition from {self.state} to {new_state.value}")
            
            logger.info(f"Found trigger: {trigger}")
            logger.info(f"Executing transition with trigger: {trigger}")
            
            # Execute the transition
            trigger_method = getattr(self, trigger)
            trigger_method(context=existing_context)  # Pass context to trigger method
            
            # Update context after successful transition
            self._context = existing_context
            
            # Ensure langchain context is synchronized
            langchain_context.update_order_memory(
                current_order=existing_context.get("order"),
                query=existing_context.get("query")
            )
            
        except Exception as e:
            logger.error(f"Error during transition: {str(e)}")
            # Only transition to error state if we're not already there
            if self.state != OrderState.ERROR.value:
                self.handle_error()

# Initialize state machine at module level
state_machine = OrderStateMachine() 
