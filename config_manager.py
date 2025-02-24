import json
import os
import logging

class ConfigManager:
    def __init__(self, config_file="default_config.json"):
        self.logger = logging.getLogger('ConfigManager')
        self.config_file = config_file
        self.config = self._load_default_config()

    def _load_default_config(self):
        default_config = {
            'detection_area': (0, 0, 100, 100),
            'detection_threshold': 0.8,
            'cast_key': 'f',
            'reel_key': 'r',
            'color_threshold': (200, 200, 200),
            'auto_start': False,
            'log_level': 'INFO'
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
        
        return default_config

    def save_config(self):
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            self.logger.info("Configuration saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving config: {str(e)}")

    def update_config(self, new_config):
        self.config.update(new_config)
        self.save_config()

    def get_config(self):
        return self.config
