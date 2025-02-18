from typing import Tuple, Dict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from loguru import logger

from ..models import OrderIntent
from ..config import INTENT_MODEL_CONFIG

class IntentClassifier:
    def __init__(self):
        """Initialize the intent classifier with primary and fallback models."""
        self.config = INTENT_MODEL_CONFIG
        
        # Get model names from config
        self.model_name = self.config["primary_model"]
        self.fallback_model_name = self.config["fallback_model"]
        
        # Get model-specific configs
        self.primary_config = self.config["primary_config"]
        self.fallback_config = self.config["fallback_config"]
        
        # Keywords and rules
        self.keywords = self.config["keywords"]
        self.score_adjustments = self.config["score_adjustments"]

        # Initialize primary model
        logger.info(f"Loading primary intent classification model: {self.model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.eval()

        # Initialize fallback model
        logger.info(f"Loading fallback intent classification model: {self.fallback_model_name}")
        self.fallback_model = AutoModelForSequenceClassification.from_pretrained(self.fallback_model_name)
        self.fallback_tokenizer = AutoTokenizer.from_pretrained(self.fallback_model_name)
        self.fallback_model.eval()

        # Define candidate labels for classification
        self.candidate_labels = [
            "This is a new food or drink order request",
            "This is a general inquiry about the menu or service",
            "This is an unsupported action request",
            "This is an unclear or ambiguous request"
        ]

        self.label_to_intent: Dict[str, OrderIntent] = {
            "This is a new food or drink order request": OrderIntent.NEW_ORDER,
            "This is a general inquiry about the menu or service": OrderIntent.GENERAL_INQUIRY,
            "This is an unsupported action request": OrderIntent.UNSUPPORTED_ACTION,
            "This is an unclear or ambiguous request": OrderIntent.UNKNOWN
        }

    def _normalize_score(self, score: float) -> float:
        """Normalize score to be between 0 and 1."""
        return max(0.0, min(1.0, score))

    def _adjust_scores(self, intent_scores: Dict[OrderIntent, float], text: str) -> Dict[OrderIntent, float]:
        """Apply score adjustments based on rules and keywords."""
        text = text.lower()
        
        # Get adjustment parameters
        boost_mult = self.score_adjustments["boost_multiplier"]
        reduce_mult = self.score_adjustments["reduction_multiplier"]
        max_boost = self.score_adjustments["max_boost_score"]
        
        # Rule 1: Boost general inquiry for menu-related questions
        if any(word in text for word in self.keywords["menu_inquiry"]):
            intent_scores[OrderIntent.GENERAL_INQUIRY] = self._normalize_score(
                max(intent_scores[OrderIntent.GENERAL_INQUIRY] * boost_mult, max_boost)
            )
        
        # Rule 2: Boost new order score for order action words combined with menu items
        if (any(word in text for word in self.keywords["order_actions"]) and 
            any(item in text for item in self.keywords["menu_items"])):
            intent_scores[OrderIntent.NEW_ORDER] = self._normalize_score(
                max(intent_scores[OrderIntent.NEW_ORDER] * boost_mult, max_boost)
            )
            intent_scores[OrderIntent.GENERAL_INQUIRY] = self._normalize_score(
                intent_scores[OrderIntent.GENERAL_INQUIRY] * reduce_mult
            )
        
        # Rule 3: Boost unsupported action for non-food service requests
        if any(word in text for word in self.keywords["unsupported"]):
            intent_scores[OrderIntent.UNSUPPORTED_ACTION] = self._normalize_score(
                max(intent_scores[OrderIntent.UNSUPPORTED_ACTION] * boost_mult, max_boost)
            )
            intent_scores[OrderIntent.GENERAL_INQUIRY] = self._normalize_score(
                intent_scores[OrderIntent.GENERAL_INQUIRY] * reduce_mult
            )
        
        # Rule 4: Reduce new order score if no menu items mentioned
        if not any(word in text for word in self.keywords["menu_items"]):
            intent_scores[OrderIntent.NEW_ORDER] = self._normalize_score(
                intent_scores[OrderIntent.NEW_ORDER] * self.score_adjustments["min_menu_item_confidence"]
            )
        
        # Rule 5: Boost unknown score for vague requests
        if any(phrase in text for phrase in self.keywords["vague_terms"]):
            intent_scores[OrderIntent.UNKNOWN] = self._normalize_score(
                max(intent_scores[OrderIntent.UNKNOWN] * boost_mult, max_boost)
            )
        
        return intent_scores

    def _get_model_specific_config(self, use_fallback: bool) -> dict:
        """Get the configuration for the specified model."""
        return self.fallback_config if use_fallback else self.primary_config

    def _classify_internal(self, text: str, use_fallback: bool = False) -> Tuple[OrderIntent, float]:
        """Internal classification method that can use either primary or fallback model."""
        model = self.fallback_model if use_fallback else self.model
        tokenizer = self.fallback_tokenizer if use_fallback else self.tokenizer
        model_config = self._get_model_specific_config(use_fallback)

        if use_fallback:
            # DeBERTa-specific hypotheses optimized for contradiction/entailment/neutral
            intent_patterns = model_config["intent_patterns"]
            hypotheses = [
                # NEW_ORDER - Strong entailment patterns for ordering
                (OrderIntent.NEW_ORDER, [
                    (pattern, text) for pattern in intent_patterns["new_order"]
                ]),
                # GENERAL_INQUIRY - Neutral patterns for questions
                (OrderIntent.GENERAL_INQUIRY, [
                    (pattern, text) for pattern in intent_patterns["general_inquiry"]
                ]),
                # UNSUPPORTED_ACTION - Contradiction patterns for non-food requests
                (OrderIntent.UNSUPPORTED_ACTION, [
                    (pattern, text) for pattern in intent_patterns["unsupported_action"]
                ]),
                # UNKNOWN - Mix of patterns to detect ambiguity
                (OrderIntent.UNKNOWN, [
                    (pattern, text) for pattern in intent_patterns["unknown"]
                ])
            ]

            # Calculate scores using contradiction/entailment/neutral outputs
            intent_scores = {}
            for intent, intent_hypotheses in hypotheses:
                hypothesis_scores = []
                for premise, hypothesis in intent_hypotheses:
                    inputs = tokenizer(
                        premise,
                        hypothesis,
                        padding=model_config["padding"],
                        truncation=model_config["truncation"],
                        max_length=model_config["max_length"],
                        return_tensors="pt"
                    )

                    with torch.no_grad():
                        outputs = model(**inputs)
                        # Get probabilities for contradiction (0), entailment (1), neutral (2)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
                        
                        # Use configured weights for each intent type
                        weights = model_config["score_weights"][intent.value]
                        score = (
                            weights["contradiction"] * probs[model_config["label_mapping"]["contradiction"]].item() +
                            weights["entailment"] * probs[model_config["label_mapping"]["entailment"]].item() +
                            weights["neutral"] * probs[model_config["label_mapping"]["neutral"]].item()
                        )
                        hypothesis_scores.append(score)

                # Use the maximum score among hypotheses for this intent
                intent_scores[intent] = max(hypothesis_scores)

        else:
            # BART-specific hypotheses and scoring
            template = model_config["hypothesis_template"]
            intent_patterns = model_config["intent_patterns"]
            
            hypotheses = [
                # NEW_ORDER - Check for explicit ordering intent with specific food/drink items
                (OrderIntent.NEW_ORDER, [
                    template.format(pattern) for pattern in intent_patterns["new_order"]
                ]),
                # GENERAL_INQUIRY - ONLY for menu/food related questions
                (OrderIntent.GENERAL_INQUIRY, [
                    template.format(pattern) for pattern in intent_patterns["general_inquiry"]
                ]),
                # UNSUPPORTED_ACTION - Any non-food service request or order management
                (OrderIntent.UNSUPPORTED_ACTION, [
                    template.format(pattern) for pattern in intent_patterns["unsupported_action"]
                ]),
                # UNKNOWN - Check for ambiguity or vague requests
                (OrderIntent.UNKNOWN, [
                    template.format(pattern) for pattern in intent_patterns["unknown"]
                ])
            ]

            # Calculate entailment scores for each intent
            intent_scores = {}
            for intent, intent_hypotheses in hypotheses:
                hypothesis_scores = []
                for hypothesis in intent_hypotheses:
                    inputs = tokenizer(
                        text,
                        hypothesis,
                        padding=model_config["padding"],
                        truncation=model_config["truncation"],
                        max_length=model_config["max_length"],
                        return_tensors="pt"
                    )

                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
                        # Use configured weights
                        weights = model_config["score_weights"]
                        score = (
                            weights["entailment"] * probs[2].item() +  # entailment
                            weights["contradiction"] * probs[0].item() +  # contradiction
                            weights["neutral"] * probs[1].item()  # neutral
                        )
                        hypothesis_scores.append(score)

                # Use the maximum score among hypotheses for this intent
                intent_scores[intent] = max(hypothesis_scores)

        # Find the best matching intent
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        return best_intent[0], self._normalize_score(best_intent[1])

    def classify(self, text: str) -> Tuple[OrderIntent, float]:
        """Classify the intent of the user's input text using zero-shot classification."""
        # Normalize input text
        text = text.strip().lower()

        # Try primary model first
        intent, confidence = self._classify_internal(text, use_fallback=False)

        # Log primary model results
        logger.info(
            f"Primary model classification:\n"
            f"Input: {text}\n"
            f"Predicted Intent: {intent.value}\n"
            f"Confidence: {confidence:.2f}"
        )

        # If confidence is low, try fallback model
        if confidence < self.primary_config["confidence_threshold"]:
            logger.warning(f"Low confidence with primary model ({confidence:.2f}). Trying fallback model.")
            fallback_intent, fallback_confidence = self._classify_internal(text, use_fallback=True)

            # Log fallback model results
            logger.info(
                f"Fallback model classification:\n"
                f"Input: {text}\n"
                f"Predicted Intent: {fallback_intent.value}\n"
                f"Confidence: {fallback_confidence:.2f}"
            )

            # Use fallback result if it has higher confidence and meets threshold
            if fallback_confidence > confidence and fallback_confidence >= self.fallback_config["confidence_threshold"]:
                logger.info(f"Using fallback model prediction (confidence: {fallback_confidence:.2f})")
                return fallback_intent, fallback_confidence

            # If both models have low confidence, return UNKNOWN
            return OrderIntent.UNKNOWN, min(confidence, fallback_confidence)

        return intent, confidence

    def request_clarification(self, text: str) -> OrderIntent:
        """Handle ambiguous user input by requesting clarification."""
        logger.warning(
            f"Low confidence classification for both models:\n"
            f"Input: {text}\n"
            f"Requesting clarification from user."
        )
        return OrderIntent.UNKNOWN

    def is_order_intent(self, text: str) -> bool:
        """Quick check if the intent is a new order."""
        intent, _ = self.classify(text)
        return intent == OrderIntent.NEW_ORDER

    def get_intent_explanation(self, text: str) -> str:
        """Get a human-readable explanation of the intent classification."""
        intent, confidence = self.classify(text)

        explanations = {
            OrderIntent.NEW_ORDER: "This appears to be a new food order request.",
            OrderIntent.GENERAL_INQUIRY: "This seems to be a general question about our menu or service.",
            OrderIntent.UNSUPPORTED_ACTION: "This request contains an action we don't support.",
            OrderIntent.UNKNOWN: "This request is ambiguous or does not match any category. Could you please clarify?"
        }

        return f"{explanations[intent]} (confidence: {confidence:.2%})"

# Initialize classifier at module level for reuse
intent_classifier = IntentClassifier()
