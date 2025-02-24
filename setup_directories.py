"""Setup script to create necessary directories for Sound Macro application"""
import os
from pathlib import Path
import logging

def setup_directories():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Create macros directory
        macros_dir = Path("macros")
        macros_dir.mkdir(exist_ok=True)
        logger.info("Created macros directory")
        
        # Create temp directory for recordings
        temp_dir = Path("temp_recordings")
        temp_dir.mkdir(exist_ok=True)
        logger.info("Created temp_recordings directory")
        
        return True
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        return False

if __name__ == "__main__":
    setup_directories()
