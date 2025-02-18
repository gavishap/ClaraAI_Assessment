from fastapi import FastAPI
from loguru import logger
import uvicorn

from .config import API_CONFIG
from .routes import orders, inquiries
from .utils.logging import setup_logging

# Setup logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title=API_CONFIG["title"],
    version=API_CONFIG["version"],
    description=API_CONFIG["description"]
)

# Include routers
app.include_router(orders.router)
app.include_router(inquiries.router)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    logger.info(f"Starting {API_CONFIG['title']} v{API_CONFIG['version']}")
    uvicorn.run(
        app,
        host=API_CONFIG["host"],
        port=API_CONFIG["port"]
    ) 
