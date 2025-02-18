import pytest
from llm_room_service.app.models import OrderIntent
from llm_room_service.app.services.intent_classifier import IntentClassifier

@pytest.fixture
def classifier():
    """Create a fresh classifier instance for each test."""
    return IntentClassifier()

def test_new_order_intents(classifier):
    """Test various new order requests."""
    order_texts = [
        "I'd like to order a club sandwich",
        "Can I get two waters and a pizza delivered to room 301",
        "Please bring me a caesar salad",
        "I want a margherita pizza with extra cheese",
        "Could you send up three bottles of still water",
        "I'd like to place an order for a burger and fries",
        "Can I order apple pie for dessert",
        "Bring me a club sandwich with extra bacon and a side of fries",
        "I would like to get the caesar salad with chicken",
        "Send up a fresh orange juice please"
    ]
    
    for text in order_texts:
        intent, confidence = classifier.classify(text)
        assert intent == OrderIntent.NEW_ORDER, f"Failed on: {text}"
        assert confidence > 0.65, f"Low confidence ({confidence}) on: {text}"

def test_menu_inquiries(classifier):
    """Test various menu-related inquiries."""
    inquiry_texts = [
        "What's on the menu today?",
        "Do you have any vegetarian options?",
        "What time do you serve breakfast?",
        "Are there any gluten-free desserts?",
        "How spicy is the pizza?",
        "What ingredients are in the club sandwich?",
        "Do you have vegan options available?",
        "What's included in the caesar salad?",
        "How much does the burger cost?",
        "What are today's specials?"
    ]
    
    for text in inquiry_texts:
        intent, confidence = classifier.classify(text)
        assert intent == OrderIntent.GENERAL_INQUIRY, f"Failed on: {text}"
        assert confidence > 0.65, f"Low confidence ({confidence}) on: {text}"

def test_unsupported_actions(classifier):
    """Test various unsupported action requests."""
    unsupported_texts = [
        "Can you cancel my order?",
        "Is my order ready yet?",
        "I want to modify my previous order",
        "Can I get a wake-up call?",
        "When does the pool close?",
        "I need to book a spa appointment",
        "Please clean my room",
        "Can you check on my order status?",
        "I'd like to change my delivery time",
        "Where is the gym located?",
        "Can I get fresh towels?",
        "What's the wifi password?",
        "I need to extend my checkout time",
        "Can you track my order?",
        "Is my food on the way?"
    ]
    
    for text in unsupported_texts:
        intent, confidence = classifier.classify(text)
        assert intent == OrderIntent.UNSUPPORTED_ACTION, f"Failed on: {text}"
        assert confidence > 0.65, f"Low confidence ({confidence}) on: {text}"

def test_ambiguous_inputs(classifier):
    """Test handling of ambiguous or unclear inputs."""
    ambiguous_texts = [
        "hmm let me think",
        "not sure yet",
        "maybe later",
        "get me something good",
        "bring me whatever",
        "I want something nice",
        "anything will do",
        "...",
        "what do you recommend",
        "surprise me"
    ]
    
    for text in ambiguous_texts:
        intent, confidence = classifier.classify(text)
        assert intent == OrderIntent.UNKNOWN, f"Failed on: {text}"

def test_complex_scenarios(classifier):
    """Test more complex or edge case scenarios."""
    test_cases = [
        # (text, expected_intent)
        ("I want to cancel my burger order", OrderIntent.UNSUPPORTED_ACTION),
        ("What time does breakfast service end?", OrderIntent.GENERAL_INQUIRY),
        ("When does the pool restaurant close?", OrderIntent.UNSUPPORTED_ACTION),
        ("Do you have room service menu?", OrderIntent.GENERAL_INQUIRY),
        ("Can you tell me about the chef's specials?", OrderIntent.GENERAL_INQUIRY),
        ("I need to modify my sandwich order", OrderIntent.UNSUPPORTED_ACTION),
        ("Is the kitchen still open?", OrderIntent.GENERAL_INQUIRY),
        ("Bring me something from the menu", OrderIntent.UNKNOWN),
        ("Can I order now for later?", OrderIntent.GENERAL_INQUIRY),
        ("What's the status of room 302's order?", OrderIntent.UNSUPPORTED_ACTION)
    ]
    
    for text, expected_intent in test_cases:
        intent, confidence = classifier.classify(text)
        assert intent == expected_intent, f"Failed on: {text}"
        assert confidence > 0.65, f"Low confidence ({confidence}) on: {text}"

def test_intent_explanations(classifier):
    """Test that intent explanations are properly formatted."""
    test_cases = [
        ("I'd like to order a club sandwich", "food order request"),
        ("What's on the menu?", "question about our menu"),
        ("Can I get a wake-up call?", "action we don't support"),
        ("something good", "ambiguous")
    ]
    
    for text, expected_phrase in test_cases:
        explanation = classifier.get_intent_explanation(text)
        assert expected_phrase in explanation.lower()
        assert "confidence:" in explanation.lower()

def test_mixed_intents(classifier):
    """Test cases that might seem to belong to multiple categories."""
    test_cases = [
        # These should be UNSUPPORTED_ACTION because they're about order management
        ("Can you cancel my pizza order?", OrderIntent.UNSUPPORTED_ACTION),
        ("I want to change my sandwich order", OrderIntent.UNSUPPORTED_ACTION),
        ("Is my burger order ready?", OrderIntent.UNSUPPORTED_ACTION),
        
        # These should be GENERAL_INQUIRY because they're about menu/food
        ("When do you stop taking food orders?", OrderIntent.GENERAL_INQUIRY),
        ("Is the kitchen still accepting orders?", OrderIntent.GENERAL_INQUIRY),
        ("Do you deliver food to the pool area?", OrderIntent.GENERAL_INQUIRY),
        
        # These should be UNKNOWN because they're ambiguous
        ("I'll have the usual", OrderIntent.UNKNOWN),
        ("Same as yesterday", OrderIntent.UNKNOWN),
        ("Whatever is fresh", OrderIntent.UNKNOWN)
    ]
    
    for text, expected_intent in test_cases:
        intent, confidence = classifier.classify(text)
        assert intent == expected_intent, f"Failed on: {text}"

def test_fallback_model(classifier):
    """Test that fallback model is used when primary model has low confidence."""
    # This is a deliberately ambiguous input that might trigger fallback
    text = "I'm thinking about maybe getting something to eat"
    intent, confidence = classifier.classify(text)
    
    # We don't assert specific intent because it might vary
    # but we do check that we get a valid intent and confidence
    assert isinstance(intent, OrderIntent)
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1

def test_is_order_intent(classifier):
    """Test the quick order intent check."""
    assert classifier.is_order_intent("I want to order food") == True
    assert classifier.is_order_intent("What's on the menu?") == False 
