from .services.intent_classifier import intent_classifier
from loguru import logger

def test_intent_interactive():
    """Interactive testing of the intent classifier."""
    print("\n=== Intent Classifier Interactive Testing ===")
    print("Type 'quit' to exit\n")
    
    while True:
        # Get user input
        text = input("\nEnter text to classify: ").strip()
        
        if text.lower() == 'quit':
            break
            
        if not text:
            continue
            
        # Classify the intent
        intent, confidence = intent_classifier.classify(text)
        
        # Get explanation
        explanation = intent_classifier.get_intent_explanation(text)
        
        # Print results
        print("\nResults:")
        print(f"Intent: {intent.value}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Explanation: {explanation}")
        print("\n" + "="*50)

if __name__ == "__main__":
    test_intent_interactive() 
