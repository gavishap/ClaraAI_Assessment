# LLM Room Service Order Management System

An advanced natural language room service order processing system that uses state-of-the-art language models and machine learning techniques to understand, validate, and process customer orders.

## Core Features

- Natural language order processing with context awareness
- Multi-stage intent classification with fallback models
- Semantic similarity for menu item matching
- Fuzzy matching for modifications and corrections
- State machine for order flow management
- Inventory validation and suggestion generation
- Mock kitchen API for order simulation

## Technical Architecture

### Intent Classification

- Primary Model: BART-large-mnli for zero-shot classification
- Fallback Model: DeBERTa-v3 for natural language inference
- Intent Categories:
  - NEW_ORDER: Food/drink ordering requests
  - GENERAL_INQUIRY: Menu and service questions
  - UNSUPPORTED_ACTION: Non-supported requests
  - UNKNOWN: Ambiguous or unclear requests

### Order Processing Pipeline

1. **Intent Classification**

   - Zero-shot classification using BART
   - Fallback to DeBERTa for low confidence cases
   - Rule-based score adjustments

2. **Order Extraction**

   - GPT-4 for structured order extraction
   - Semantic embedding search for relevant menu items
   - Fallback to rule-based parsing for edge cases

3. **Validation & Suggestions**

   - Menu item validation against available options
   - Inventory level checking
   - Modification compatibility verification
   - Smart suggestion generation using embeddings

4. **State Management**
   - LangChain for context management
   - Custom state machine for order flow
   - States:
     - INITIAL
     - INTENT_CLASSIFICATION
     - MENU_INQUIRY
     - ORDER_EXTRACTION
     - ITEM_VALIDATION
     - ITEM_SELECTION
     - MODIFICATION_VALIDATION
     - MODIFICATION_SELECTION
     - QUANTITY_VALIDATION
     - ORDER_CONFIRMATION
     - ORDER_COMPLETED
     - ERROR

### Key Components

#### Embedding Service

- Uses Sentence Transformers (all-MiniLM-L6-v2)
- Menu item embeddings for semantic search
- Modification matching and suggestions
- Caching for performance optimization

#### LangChain Context Manager

- Conversation history tracking
- Order memory management
- State-specific prompting
- Context preservation across interactions

#### Validation System

- Multi-stage validation pipeline
- Menu item verification
- Inventory level checking
- Modification compatibility
- Room number validation
- Special instruction validation

#### Fuzzy Matching

- String similarity for item matching
- Modification matching
- Quantity extraction
- Error tolerance in user inputs

#### Mock Kitchen API

- Order ID generation
- Order status tracking
- Kitchen queue simulation
- Order modification handling

## Running the System

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the interactive test pipeline:

```bash
python -m llm_room_service.app.test_order_pipeline
```

## Example Interactions

1. New Order:

```
Enter order text: I'd like a caesar salad with chicken and two bottles of still water
```

2. Menu Inquiry:

```
Enter order text: What ingredients are in the club sandwich?
```

3. Modification Selection:

```
Enter order text: Make that salad with extra parmesan
```

## Advanced Features

### Reprompting System

- Automatic error recovery
- Context-aware reprompting
- Structured output repair
- Validation failure handling

### Smart Suggestion Engine

- Semantic similarity-based suggestions
- Category-aware recommendations
- Inventory-aware alternatives
- Modification compatibility checking

### Context Management

- Persistent order context
- Conversation history tracking
- State-aware prompting
- Memory management for long interactions

### Validation Pipeline

1. Menu Validation
   - Item existence
   - Category verification
   - Modification compatibility
2. Inventory Validation

   - Stock level checking
   - Alternative suggestion generation
   - Quantity adjustment recommendations

3. Order Validation
   - Room number verification
   - Special instruction validation
   - Total order validation

## Error Handling

- Graceful degradation with fallback models
- Multi-stage validation
- Structured error messages
- User-friendly suggestions
- Context preservation during errors

## Data Management

- In-memory order tracking
- Mock kitchen queue
- Inventory management
- Menu data structure
- Order history tracking

## Future Enhancements

- Real-time inventory updates
- Payment processing integration
- Order analytics
- Kitchen load balancing
- Delivery time estimation
- Multi-language support

## Dependencies

See `requirements.txt` for full list of dependencies, including:

- transformers
- pydantic
- langchain
- sentence-transformers
- numpy
- loguru
- python-dotenv
