from typing import Dict, List, Any, Optional
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel
from loguru import logger

class OrderMemory(BaseModel):
    """Model for storing order-specific memory."""
    original_request: str
    current_order: Optional[Dict] = None
    modifications: List[Dict] = []
    validation_issues: List[Dict] = []
    suggestions: List[Dict] = []
    user_responses: List[Dict] = []
    current_query: Optional[Dict] = None
    query: Optional[Dict] = None  # Add this field to store the active query

    def update_query(self, query: Optional[Dict]) -> None:
        """Update both query fields to ensure consistency."""
        logger.info(f"Updating query in OrderMemory: {query}")
        if query:
            # Ensure the query has all required fields
            if isinstance(query, dict):
                if 'item' in query and 'suggestions' in query:
                    if 'type' not in query:
                        query['type'] = 'item_replacement'  # Default type
                    self.current_query = query
                    self.query = query
                    logger.info(f"Query updated successfully: {query}")
                else:
                    logger.warning("Query missing required fields (item or suggestions)")
            else:
                logger.warning("Query is not a dictionary")
        else:
            logger.info("Clearing query in OrderMemory")
            self.current_query = None
            self.query = None

    def get_active_query(self) -> Optional[Dict]:
        """Get the currently active query."""
        return self.query or self.current_query

    def clear_query(self) -> None:
        """Clear both query fields."""
        self.current_query = None
        self.query = None
        logger.info("Cleared all query fields")

class LangchainContextManager:
    def __init__(self):
        """Initialize the Langchain context manager."""
        # Initialize conversation memory
        self._memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Initialize order-specific memory
        self.order_memory: Optional[OrderMemory] = None
        
        # System prompts for different states
        self._state_prompts = {
            "initial": "You are a helpful room service assistant. Your goal is to help customers place orders or answer questions about the menu.",
            "menu_inquiry": "You are answering questions about the menu. Focus on providing accurate information about items, ingredients, and modifications.",
            "order_extraction": "You are processing a food order. Focus on extracting specific items, quantities, and modifications.",
            "validation": "You are validating order details. Focus on confirming items exist and modifications are valid.",
            "suggestion": "You are providing suggestions based on the customer's preferences and available options.",
            "intent_classification": "You are classifying the intent of the user's request. Focus on determining if this is a new order, menu inquiry, or other request."
        }
        
    def start_new_conversation(self, system_prompt: str = None) -> None:
        """Start a new conversation with an optional system prompt."""
        self._memory.clear()
        if system_prompt:
            self._memory.chat_memory.add_message(
                SystemMessage(content=system_prompt)
            )
        self.order_memory = None
        
    def start_new_order(self, text: str) -> None:
        """Start tracking a new order."""
        self.order_memory = OrderMemory(
            original_request=text
        )
        
    def add_user_message(self, text: str) -> None:
        """Add a user message to the conversation history."""
        self._memory.chat_memory.add_message(
            HumanMessage(content=text)
        )
        
    def add_assistant_message(self, text: str) -> None:
        """Add an assistant message to the conversation history."""
        self._memory.chat_memory.add_message(
            AIMessage(content=text)
        )
        
    def update_order_memory(
        self,
        current_order: Optional[Dict] = None,
        modification: Optional[Dict] = None,
        validation_issue: Optional[Dict] = None,
        suggestion: Optional[Dict] = None,
        user_response: Optional[Dict] = None,
        query: Optional[Dict] = None
    ) -> None:
        """Update order memory with new information."""
        if not self.order_memory:
            logger.warning("No active order memory")
            return
            
        if current_order:
            logger.info(f"Updating current order: {current_order}")
            self.order_memory.current_order = current_order
            
        if modification:
            logger.info(f"Adding modification: {modification}")
            self.order_memory.modifications.append(modification)
            
        if validation_issue:
            logger.info(f"Adding validation issue: {validation_issue}")
            self.order_memory.validation_issues.append(validation_issue)
            
        if suggestion:
            logger.info(f"Adding suggestion: {suggestion}")
            self.order_memory.suggestions.append(suggestion)
            
        if user_response:
            logger.info(f"Adding user response: {user_response}")
            self.order_memory.user_responses.append(user_response)
            
        if query is not None:  # Allow explicit None to clear query
            logger.info(f"Updating query: {query}")
            self.order_memory.update_query(query)
            
        # Log the updated state
        logger.info(f"Updated order memory state:")
        logger.info(f"- Current order: {self.order_memory.current_order}")
        logger.info(f"- Current query: {self.order_memory.current_query}")
        logger.info(f"- Active query: {self.order_memory.query}")
        logger.info(f"- Recent suggestions: {self.order_memory.suggestions[-5:] if self.order_memory.suggestions else []}")
        
    def get_conversation_history(self) -> List[Dict]:
        """Get the full conversation history."""
        return [
            {
                "role": msg.type,
                "content": msg.content
            }
            for msg in self._memory.chat_memory.messages
        ]
        
    def get_recent_messages(self, k: int = 5) -> List[Dict]:
        """Get the k most recent messages."""
        messages = self._memory.chat_memory.messages
        return [
            {
                "role": msg.type,
                "content": msg.content
            }
            for msg in messages[-k:]
        ]
        
    def get_order_context(self) -> Dict:
        """Get the current order context."""
        if not self.order_memory:
            logger.warning("No active order memory when getting context")
            return {}
            
        context = {
            "original_request": self.order_memory.original_request,
            "current_order": self.order_memory.current_order,
            "recent_modifications": self.order_memory.modifications[-5:] if self.order_memory.modifications else [],
            "recent_issues": self.order_memory.validation_issues[-5:] if self.order_memory.validation_issues else [],
            "recent_suggestions": self.order_memory.suggestions[-5:] if self.order_memory.suggestions else [],
            "recent_responses": self.order_memory.user_responses[-5:] if self.order_memory.user_responses else [],
            "query": self.order_memory.get_active_query()  # Use the new getter method
        }
        
        logger.info(f"Retrieved order context: {context}")
        return context
        
    def clear_order_memory(self) -> None:
        """Clear the current order memory."""
        self.order_memory = None
        
    def set_state_prompt(self, state: str) -> None:
        """Set the system prompt based on the current state."""
        if state in self._state_prompts:
            self._memory.chat_memory.add_message(
                SystemMessage(content=self._state_prompts[state])
            )
            
    def get_formatted_context(self) -> str:
        """Get a formatted string of the current context for LLM prompts."""
        context = []
        
        # Add conversation history
        recent_messages = self.get_recent_messages()
        if recent_messages:
            context.append("Recent Conversation:")
            for msg in recent_messages:
                context.append(f"{msg['role'].title()}: {msg['content']}")
                
        # Add order context if exists
        order_context = self.get_order_context()
        if order_context:
            context.append("\nOrder Context:")
            if order_context["current_order"]:
                context.append("Current Order:")
                for item in order_context["current_order"]["items"]:
                    mods = f" with {', '.join(item['modifications'])}" if item['modifications'] else ""
                    context.append(f"- {item['quantity']}x {item['name']}{mods}")
                    
            if order_context["recent_issues"]:
                context.append("\nRecent Issues:")
                for issue in order_context["recent_issues"]:
                    context.append(f"- {issue['message']}")
                    
            if order_context["recent_suggestions"]:
                context.append("\nRecent Suggestions:")
                for suggestion in order_context["recent_suggestions"]:
                    if "text" in suggestion:
                        context.append(f"- {suggestion['text']}")
                    elif "item" in suggestion and "suggestions" in suggestion:
                        suggestions_text = ", ".join(f"{name} ({score:.2f})" for name, score in suggestion["suggestions"])
                        context.append(f"- For '{suggestion['item']}': {suggestions_text}")
                    
        return "\n".join(context)

# Initialize context manager at module level
langchain_context = LangchainContextManager() 
