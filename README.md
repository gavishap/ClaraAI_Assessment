# LLM Room Service Order Management System

A Python-based system that processes room service orders via natural language input, validates them against an in-memory menu and inventory, and submits confirmed orders to a mock API.

## Features

- Natural language order processing using state-of-the-art LLMs
- Intent classification using BART for zero-shot classification
- Structured order extraction with T5 and fallback rule-based parsing
- Fuzzy matching for menu items and modifications
- Semantic similarity using sentence transformers
- Comprehensive inventory management
- RESTful API with separate order and inquiry endpoints
- Robust error handling and detailed logging
- Configurable system settings

## Technical Stack

- Python 3.8+
- FastAPI for the web API
- Hugging Face Transformers (BART & T5) for LLM inference
- Sentence Transformers for semantic similarity
- Pydantic for data validation
- Loguru for structured logging

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd llm-room-service
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the FastAPI server:

```bash
python -m llm_room_service.app.main
```

The server will start on `http://localhost:8000`

## API Endpoints

### Orders

#### POST /orders/

Process a natural language room service order.

Request body:

```json
{
  "text": "I'd like a club sandwich with extra bacon and two waters to room 312",
  "room_number": 312
}
```

#### GET /orders/status/{order_id}

Get the status of an existing order.

#### GET /orders/history

Get order history with optional room number filter.

#### POST /orders/{order_id}/cancel

Cancel an existing order.

### Inquiries

#### GET /inquiries/menu

Get the current menu items with optional category filter.

#### GET /inquiries/menu/available

Get menu items that are currently in stock.

#### GET /inquiries/menu/categories

Get list of available menu categories.

#### GET /inquiries/menu/items/{item_name}

Get detailed information about a specific menu item.

#### POST /inquiries/classify

Classify a natural language inquiry.

## Project Structure

```
llm_room_service/
│── app/                        # Main application folder
│   ├── models.py               # Pydantic models for validation
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Configurations
│   ├── services/               # Business logic layer
│   │   ├── intent_classifier.py  # Intent classification
│   │   ├── order_extraction.py   # Order extraction
│   │   ├── order_validation.py   # Order validation
│   │   ├── order_processing.py   # Order processing
│   │   ├── menu_loader.py        # Menu management
│   ├── routes/                 # API routes
│   │   ├── orders.py            # Order endpoints
│   │   ├── inquiries.py         # Inquiry endpoints
│   ├── utils/                  # Helper functions
│   │   ├── logging.py           # Logging setup
│   │   ├── fuzzy_matching.py    # String matching
│   │   ├── embeddings.py        # Semantic similarity
│── data/                       # Data files
│   ├── menu.json               # Menu items
│   ├── inventory.json          # Inventory levels
```

## Error Handling

The system handles various error cases:

- Invalid/malformed LLM output with fallback to rule-based parsing
- Fuzzy matching for unrecognized menu items
- Semantic similarity for finding similar items
- Out-of-stock items with alternative suggestions
- Invalid room numbers and order validation
- Unsupported modifications with available options

## Features in Development

- Order status tracking and persistence
- Authentication and rate limiting
- Multi-language support
- Payment system integration
- Real-time inventory updates
- Order analytics and reporting
- WebSocket support for real-time updates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Testing

Run the test suite:

```bash
pytest llm_room_service/app/tests/
```

## License

MIT License
