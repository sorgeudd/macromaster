"""Setup script to create necessary directories and copy map files"""
import os
import logging
import shutil
from pathlib import Path

def setup_directories():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Create maps directory
        maps_dir = Path("maps")
        maps_dir.mkdir(exist_ok=True)
        logger.info("Created maps directory")

        # Copy map image to maps directory
        source_map = Path("attached_assets/image_1740466625160.png")
        if source_map.exists():
            dest_map = maps_dir / "default_map.png"
            shutil.copy(source_map, dest_map)
            logger.info("Copied default map to maps directory")

            # Create template maps for different terrains
            for template_name in ["forest", "mountain", "swamp"]:
                template_path = maps_dir / f"{template_name}_template.png"
                if not template_path.exists():
                    # Create a blank template file
                    shutil.copy(source_map, template_path)
                    logger.info(f"Created {template_name} template")
        else:
            logger.warning("Map source file not found")

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