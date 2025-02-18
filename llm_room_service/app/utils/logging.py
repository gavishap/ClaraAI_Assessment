import sys
from loguru import logger
from pathlib import Path

def setup_logging(log_file: Path = None):
    """Configure logging for the application."""
    # Remove default handler
    logger.remove()
    
    # Add console handler with custom format
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file handler if log file is specified
    if log_file:
        logger.add(
            log_file,
            rotation="500 MB",
            retention="10 days",
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG"
        )
    
    return logger

def log_order_request(room_number: int, text: str):
    """Log an incoming order request."""
    logger.info(f"Order request received - Room: {room_number}, Text: {text}")

def log_order_response(order_id: str, status: str, total_price: float):
    """Log an order response."""
    logger.info(f"Order processed - ID: {order_id}, Status: {status}, Total: ${total_price:.2f}")

def log_error(error_type: str, details: str):
    """Log an error with details."""
    logger.error(f"{error_type} - {details}") 
