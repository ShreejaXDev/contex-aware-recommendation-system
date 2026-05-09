"""
Context-Aware Recommendation System - Main Entry Point

This module initializes and runs the FastAPI application for serving
context-aware recommendations.
"""

import os
import sys
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the recommendation system."""
    
    try:
        import uvicorn
        from src.api.app import create_app
        
        # Create FastAPI application
        app = create_app()
        
        # Get configuration
        host = os.getenv("API_HOST", "0.0.0.0")
        port = int(os.getenv("API_PORT", 8000))
        debug = os.getenv("DEBUG", "False").lower() == "true"
        
        logger.info(f"Starting Context-Aware Recommendation System API")
        logger.info(f"Server running on {host}:{port}")
        
        # Run server
        uvicorn.run(
            app,
            host=host,
            port=port,
            debug=debug,
            log_level="info"
        )
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
