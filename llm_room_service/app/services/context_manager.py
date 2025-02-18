from typing import List, Dict, Optional
from pydantic import BaseModel
from datetime import datetime
import json
from pathlib import Path

class OrderContext(BaseModel):
    """Single order context including original and modified requests."""
    original_text: str
    current_text: str
    timestamp: datetime
    modifications: List[Dict[str, str]]  # List of modifications made to the order
    extracted_orders: List[Dict]  # History of extracted orders
    validation_issues: List[Dict]  # History of validation issues
    user_responses: List[Dict]  # History of user responses to queries

class ContextManager:
    def __init__(self, context_file: str = "order_context.json"):
        self.context_file = Path(context_file)
        self.current_context: Optional[OrderContext] = None
        self.context_history: List[OrderContext] = []
        self._load_context()

    def _load_context(self) -> None:
        """Load context history from file."""
        if self.context_file.exists():
            try:
                with open(self.context_file, 'r') as f:
                    data = json.load(f)
                    self.context_history = [OrderContext(**ctx) for ctx in data]
            except Exception as e:
                logger.error(f"Error loading context: {e}")
                self.context_history = []

    def _save_context(self) -> None:
        """Save context history to file."""
        try:
            with open(self.context_file, 'w') as f:
                json.dump(
                    [ctx.model_dump() for ctx in self.context_history],
                    f,
                    indent=2,
                    default=str
                )
        except Exception as e:
            logger.error(f"Error saving context: {e}")

    def start_new_context(self, text: str) -> OrderContext:
        """Start a new order context."""
        self.current_context = OrderContext(
            original_text=text,
            current_text=text,
            timestamp=datetime.now(),
            modifications=[],
            extracted_orders=[],
            validation_issues=[],
            user_responses=[]
        )
        self.context_history.append(self.current_context)
        self._save_context()
        return self.current_context

    def update_current_context(self, 
                             modified_text: Optional[str] = None,
                             extracted_order: Optional[Dict] = None,
                             validation_issue: Optional[Dict] = None,
                             user_response: Optional[Dict] = None) -> None:
        """Update the current context with new information."""
        if not self.current_context:
            raise ValueError("No active context")

        if modified_text:
            self.current_context.current_text = modified_text
            self.current_context.modifications.append({
                "timestamp": datetime.now().isoformat(),
                "previous": self.current_context.current_text,
                "new": modified_text
            })

        if extracted_order:
            self.current_context.extracted_orders.append({
                "timestamp": datetime.now().isoformat(),
                "order": extracted_order
            })

        if validation_issue:
            self.current_context.validation_issues.append({
                "timestamp": datetime.now().isoformat(),
                "issue": validation_issue
            })

        if user_response:
            self.current_context.user_responses.append({
                "timestamp": datetime.now().isoformat(),
                "response": user_response
            })

        self._save_context()

    def get_context_summary(self) -> Dict:
        """Get a summary of the current context for prompting."""
        if not self.current_context:
            return {}

        return {
            "original_request": self.current_context.original_text,
            "current_request": self.current_context.current_text,
            "modifications_made": [mod["new"] for mod in self.current_context.modifications[-3:]],  # Last 3 modifications
            "recent_issues": [issue["issue"] for issue in self.current_context.validation_issues[-3:]],  # Last 3 issues
            "user_responses": [resp["response"] for resp in self.current_context.user_responses[-3:]]  # Last 3 responses
        }

# Initialize context manager at module level
context_manager = ContextManager() 
