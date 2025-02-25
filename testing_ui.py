"""Real-time testing interface for the fishing bot systems"""
import logging
import json
import os
import platform
import time
import traceback
from pathlib import Path
import atexit
import socket

try:
    from flask import Flask, render_template, jsonify
    from flask_sock import Sock
except ImportError as e:
    print(f"Failed to import Flask dependencies: {e}")
    print("Please ensure flask and flask-sock are installed")
    raise

try:
    from map_manager import MapManager
    from mock_environment import MockEnvironment
    from sound_macro_manager import SoundMacroManager
except ImportError as e:
    print(f"Failed to import local modules: {e}")
    print("Please ensure all required modules are available")
    raise

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('testing_ui.log')
    ]
)
logger = logging.getLogger('TestingUI')

try:
    # Initialize Flask and WebSocket
    app = Flask(__name__)
    sock = Sock(app)
except Exception as e:
    logger.error(f"Failed to initialize Flask app: {e}")
    logger.error(traceback.format_exc())
    raise

# Global TestingUI instance
testing_ui = None

class TestingUI:
    def __init__(self):
        self.logger = logger
        self.initialized = False
        try:
            # Initialize components
            self.map_manager = MapManager()
            self.mock_env = MockEnvironment()
            self.sound_manager = SoundMacroManager(test_mode=(platform.system() != 'Windows'))

            # Ensure macros directory exists
            self.macro_dir = Path('macros')
            self.macro_dir.mkdir(exist_ok=True)
            self.logger.info(f"Created macros directory at {self.macro_dir}")

            # Create test macro if it doesn't exist
            self._create_test_macro()

            self.initialized = True
            self.logger.info("Testing UI initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing TestingUI: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.mock_env and self.mock_env.is_running:
                self.mock_env.stop_simulation()
            if self.sound_manager:
                self.sound_manager.stop_monitoring()
            self.logger.info("Resources cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def _create_test_macro(self):
        """Create a test macro file if it doesn't exist"""
        test_macro_path = self.macro_dir / 'test_macro.json'
        if not test_macro_path.exists():
            test_macro = {
                "name": "test_macro",
                "actions": [
                    {
                        "type": "move",
                        "x": 100,
                        "y": 100,
                        "button": "left",
                        "duration": 0.1,
                        "timestamp": 0
                    }
                ],
                "created": 0
            }
            with open(test_macro_path, 'w') as f:
                json.dump(test_macro, f, indent=2)
            self.logger.info(f"Created test macro at {test_macro_path}")

def cleanup_resources():
    """Cleanup resources"""
    global testing_ui
    try:
        if testing_ui:
            if testing_ui.mock_env and testing_ui.mock_env.is_running:
                testing_ui.mock_env.stop_simulation()
            if testing_ui.sound_manager:
                testing_ui.sound_manager.stop_monitoring()
            logger.info("Resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def run():
    """Start the server"""
    try:
        global testing_ui
        logger.info("Initializing Testing UI server...")
        testing_ui = TestingUI()

        if not testing_ui.initialized:
            logger.error("Failed to initialize Testing UI")
            return

        logger.info("Starting Testing UI server on port 5002")
        try:
            # Test if port is available before starting
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('0.0.0.0', 5002))
            sock.close()
            logger.info("Port 5002 is available")

            # Use threaded=True to handle WebSocket connections properly
            app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)
        except OSError as e:
            logger.error(f"Port 5002 is not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Error starting Flask server: {e}")
            logger.error(traceback.format_exc())
            raise
    except Exception as e:
        logger.error(f"Server error: {e}")
        logger.error(traceback.format_exc())
        cleanup_resources()
        raise
    finally:
        cleanup_resources()

@app.route('/')
def index():
    """Flask route for web interface"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/sounds')
def get_sounds():
    """API endpoint for getting available sound triggers"""
    global testing_ui
    try:
        if not testing_ui or not testing_ui.initialized:
            return jsonify({'error': 'Server not initialized', 'sounds': []}), 500

        sounds = testing_ui.sound_manager.sound_trigger.get_trigger_names()
        testing_ui.logger.info(f"Found {len(sounds)} sound triggers: {sounds}")

        response = jsonify({'sounds': sounds})
        response.headers['Content-Type'] = 'application/json'
        response.headers['Cache-Control'] = 'no-store'
        return response

    except Exception as e:
        logger.error(f"Error loading sounds: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'sounds': []}), 500

@app.route('/macros')
def get_macros():
    """API endpoint for getting available macros"""
    global testing_ui
    try:
        if not testing_ui or not testing_ui.initialized:
            return jsonify({'error': 'Server not initialized', 'macros': []}), 500

        macro_files = [f.stem for f in testing_ui.macro_dir.glob('*.json')]
        testing_ui.logger.info(f"Found {len(macro_files)} macros: {macro_files}")

        response = jsonify({'macros': macro_files})
        response.headers['Content-Type'] = 'application/json'
        response.headers['Cache-Control'] = 'no-store'
        return response

    except Exception as e:
        logger.error(f"Error loading macros: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'macros': []}), 500

@sock.route('/ws')
def websocket(ws):
    """WebSocket endpoint for real-time updates and commands"""
    global testing_ui
    try:
        if not testing_ui or not testing_ui.initialized:
            ws.send(json.dumps({
                'type': 'error',
                'message': 'Server not initialized'
            }))
            return

        logger.info("New WebSocket connection established")
        ws.send(json.dumps({
            'type': 'status_update',
            'window_status': 'Not Detected',
            'bot_status': 'Stopped',
            'learning_status': 'Inactive',
            'macro_status': 'Ready',
            'sound_status': 'Ready',
            'monitoring_status': 'Stopped'
        }))

        while True:
            try:
                message = ws.receive()
                if not message:
                    continue

                data = json.loads(message)
                testing_ui.logger.info(f"Received command: {data['type']}")

                # Handle WebSocket messages here
                if data['type'] == 'start_sound_recording':
                    handle_start_sound_recording(ws, data)
                elif data['type'] == 'stop_sound_recording':
                    handle_stop_sound_recording(ws)
                elif data['type'] == 'start_macro_recording':
                    handle_start_macro_recording(ws, data)
                elif data['type'] == 'stop_macro_recording':
                    handle_stop_macro_recording(ws)

            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                logger.error(traceback.format_exc())
                ws.send(json.dumps({
                    'type': 'error',
                    'message': f'Error processing command: {str(e)}'
                }))

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        logger.error(traceback.format_exc())

def handle_start_sound_recording(ws, data):
    """Handle start sound recording command"""
    sound_name = data.get('sound_name', '')
    if not sound_name:
        ws.send(json.dumps({
            'type': 'error',
            'message': 'Please enter a sound trigger name'
        }))
        return

    if testing_ui.sound_manager.record_sound_trigger(sound_name):
        ws.send(json.dumps({
            'type': 'log',
            'level': 'info',
            'message': f'Recording started: {sound_name}'
        }))

def handle_stop_sound_recording(ws):
    """Handle stop sound recording command"""
    if testing_ui.sound_manager.stop_recording():
        ws.send(json.dumps({
            'type': 'recording_complete',
            'recording_type': 'sound'
        }))

def handle_start_macro_recording(ws, data):
    """Handle start macro recording command"""
    macro_name = data.get('macro_name', '')
    if testing_ui.sound_manager.record_macro(macro_name):
        ws.send(json.dumps({
            'type': 'log',
            'level': 'info',
            'message': f'Started recording macro: {macro_name}'
        }))

def handle_stop_macro_recording(ws):
    """Handle stop macro recording command"""
    if testing_ui.sound_manager.stop_macro_recording():
        ws.send(json.dumps({
            'type': 'recording_complete',
            'recording_type': 'macro'
        }))

if __name__ == "__main__":
    # Register cleanup on exit
    atexit.register(cleanup_resources)
    run()