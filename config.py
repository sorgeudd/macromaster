"""Configuration settings for the fishing bot"""
from pathlib import Path

# Key bindings
CAST_KEY = 'f3'  # Key to cast/reel fishing line
STOP_KEY = 'f6'  # Emergency stop key

# Screen capture settings
MONITOR = {
    "top": 0,       # Distance from the top of the screen
    "left": 0,      # Distance from the left of the screen
    "width": 800,   # Width of capture area
    "height": 600   # Height of capture area
}

# Color detection settings
FISHING_COLORS = {
    'blue': {
        'lower': [100, 150, 150],  # HSV lower bound
        'upper': [140, 255, 255]   # HSV upper bound
    }
}

# Detection settings
BITE_THRESHOLD = 100  # Minimum blue pixels for bite detection
MIN_BITE_DURATION = 0.2  # Minimum time (seconds) the bite should be visible

# Timing settings
CAST_DELAY = 2.0         # Delay after casting
RECAST_DELAY = 1.0       # Delay after reeling in
DETECTION_INTERVAL = 0.1  # Time between detection checks

# Debug settings
DEBUG_SCREENSHOTS = True
SCREENSHOT_DIR = Path('debug_screenshots')

# Logging settings
LOG_LEVEL = 'INFO'  # Can be DEBUG, INFO, WARNING, ERROR, CRITICAL