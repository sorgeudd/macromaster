"""Real-time testing interface for macro systems"""
import logging
import json
import os
import time
import traceback
import socket
from datetime import datetime
from pathlib import Path
from PIL import ImageGrab

try:
    from flask import Flask, render_template, send_file, jsonify
    from flask_sock import Sock
except ImportError as e:
    print(f"Failed to import Flask dependencies: {e}")
    print("Please ensure flask and flask-sock are installed")
    raise

try:
    from sound_macro_manager import SoundMacroManager
except ImportError as e:
    print(f"Failed to import local modules: {e}")
    print("Please ensure all required modules are available")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('testing_ui.log')
    ]
)
logger = logging.getLogger('TestingUI')

# Initialize Flask and WebSocket
app = Flask(__name__)
sock = Sock(app)

# Enable WebSocket support through proxy
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}

# Global TestingUI instance
testing_ui = None

def verify_port_available(port):
    """Check if a port is available"""
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.settimeout(1)
        test_socket.bind(('0.0.0.0', port))
        test_socket.close()
        return True
    except OSError:
        return False

class TestingUI:
    def __init__(self):
        self.logger = logger
        self.initialized = False
        try:
            # Initialize sound macro manager
            self.sound_manager = SoundMacroManager(test_mode=True)

            # Ensure directories exist
            self.macro_dir = Path('macros')
            self.macro_dir.mkdir(exist_ok=True)
            self.logger.info(f"Created macros directory at {self.macro_dir}")

            self.screenshots_dir = Path('debug_screenshots')
            self.screenshots_dir.mkdir(exist_ok=True)
            self.logger.info(f"Created screenshots directory at {self.screenshots_dir}")

            # Create test macro if it doesn't exist
            self._create_test_macro()

            self.initialized = True
            self.logger.info("Testing UI initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing TestingUI: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def take_screenshot(self, macro_name: str = None) -> tuple[bool, str, str]:
        """Take a screenshot and save it with timestamp"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{macro_name}_{timestamp}.png" if macro_name else f"screenshot_{timestamp}.png"
            filepath = self.screenshots_dir / filename

            # Capture the screen
            screenshot = ImageGrab.grab()
            screenshot.save(str(filepath))

            self.logger.info(f"Screenshot saved: {filepath}")
            return True, str(filepath), filename
        except Exception as e:
            self.logger.error(f"Error taking screenshot: {e}")
            self.logger.error(traceback.format_exc())
            return False, "", str(e)

    def cleanup(self):
        """Cleanup resources"""
        try:
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
                "hotkey": None,
                "actions": []
            }
            with open(test_macro_path, 'w') as f:
                json.dump(test_macro, f, indent=2)
            self.logger.info(f"Created test macro at {test_macro_path}")

@app.route('/')
def index():
    """Flask route for web interface"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {e}")
        logger.error(traceback.format_exc())
        return f"Error: {str(e)}", 500

@sock.route('/ws')
def websocket(ws):
    """WebSocket endpoint"""
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
            'macro_status': 'Ready',
            'sound_status': 'Ready'
        }))

        while True:
            try:
                message = ws.receive()
                if not message:
                    logger.info("Client disconnected")
                    break

                data = json.loads(message)
                testing_ui.logger.info(f"Received command: {data['type']}")

                if data['type'] == 'ping':
                    ws.send(json.dumps({'type': 'pong'}))
                    continue

                if data['type'] == 'take_screenshot':
                    handle_take_screenshot(ws, data)
                elif data['type'] == 'assign_hotkey':
                    handle_assign_hotkey(ws, data)
                elif data['type'] == 'clear_hotkey':
                    handle_clear_hotkey(ws, data)
                elif data['type'] == 'start_macro_recording':
                    handle_start_macro_recording(ws, data)
                elif data['type'] == 'stop_macro_recording':
                    handle_stop_macro_recording(ws)
                elif data['type'] == 'start_sound_recording':
                    handle_start_sound_recording(ws, data)
                elif data['type'] == 'stop_sound_recording':
                    handle_stop_sound_recording(ws)

            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
                try:
                    ws.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid message format'
                    }))
                except Exception:
                    logger.error("Failed to send error response")
                    break

            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                logger.error(traceback.format_exc())
                try:
                    ws.send(json.dumps({
                        'type': 'error',
                        'message': f'Error processing command: {str(e)}'
                    }))
                except Exception:
                    logger.error("Failed to send error message to client")
                    break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        logger.error(traceback.format_exc())

def handle_take_screenshot(ws, data):
    """Handle screenshot capture request"""
    macro_name = data.get('macro_name')
    success, filepath, filename = testing_ui.take_screenshot(macro_name)
    if success:
        ws.send(json.dumps({
            'type': 'screenshot_taken',
            'filepath': filepath,
            'filename': filename
        }))
    else:
        ws.send(json.dumps({
            'type': 'error',
            'message': f'Failed to take screenshot: {filename}'
        }))

def handle_assign_hotkey(ws, data):
    """Handle hotkey assignment command"""
    macro_name = data.get('macro_name', '')
    hotkey = data.get('hotkey', '')

    if not macro_name or not hotkey:
        ws.send(json.dumps({
            'type': 'error',
            'message': 'Please provide both macro name and hotkey'
        }))
        return

    if testing_ui.sound_manager.assign_hotkey(macro_name, hotkey):
        ws.send(json.dumps({
            'type': 'log',
            'level': 'info',
            'message': f'Assigned hotkey {hotkey} to macro {macro_name}'
        }))
    else:
        ws.send(json.dumps({
            'type': 'error',
            'message': 'Failed to assign hotkey'
        }))

def handle_clear_hotkey(ws, data):
    """Handle clear hotkey command"""
    macro_name = data.get('macro_name', '')

    if not macro_name:
        ws.send(json.dumps({
            'type': 'error',
            'message': 'Please provide a macro name'
        }))
        return

    if testing_ui.sound_manager.remove_hotkey(macro_name):
        ws.send(json.dumps({
            'type': 'log',
            'level': 'info',
            'message': f'Cleared hotkey for macro {macro_name}'
        }))
    else:
        ws.send(json.dumps({
            'type': 'error',
            'message': 'No hotkey assigned to clear'
        }))

def handle_start_macro_recording(ws, data):
    """Handle start macro recording command"""
    macro_name = data.get('macro_name', '')
    if not macro_name:
        ws.send(json.dumps({
            'type': 'error',
            'message': 'Please enter a macro name'
        }))
        return

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

@app.route('/screenshots/<path:filename>')
def serve_screenshot(filename):
    """Serve screenshot files"""
    global testing_ui
    try:
        return send_file(str(testing_ui.screenshots_dir / filename))
    except Exception as e:
        logger.error(f"Error serving screenshot: {e}")
        return f"Error: {str(e)}", 500

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

def cleanup_resources():
    """Cleanup resources"""
    global testing_ui
    try:
        if testing_ui:
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
            raise RuntimeError("Failed to initialize Testing UI")

        if not verify_port_available(5000):
            logger.error("Port 5000 is not available")
            raise RuntimeError("Port 5000 is not available")

        logger.info("Starting Testing UI server on port 5000")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

    except Exception as e:
        logger.error(f"Server error: {e}")
        logger.error(traceback.format_exc())
        cleanup_resources()
        raise

if __name__ == "__main__":
    run()