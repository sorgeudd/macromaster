"""Testing UI interface for Sound Macro Recorder application"""
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TestingUI')

# Initialize Flask and WebSocket
app = Flask(__name__, static_folder='static')
sock = Sock(app)

# Enable WebSocket support through proxy
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}

# Global TestingUI instance
testing_ui = None

def verify_port_available(port):
    """Check if a port is available"""
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        sock.bind(('0.0.0.0', port))
        return True
    except OSError:
        logger.error(f"Port {port} is not available")
        return False
    finally:
        if sock:
            sock.close()

def cleanup_existing_server():
    """Cleanup any existing server resources"""
    try:
        # Try to connect to check if server is running
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('0.0.0.0', 5000))
        sock.close()

        if result == 0:
            logger.info("Found existing server, waiting for port to be released")
            time.sleep(2)  # Wait for cleanup
    except Exception as e:
        logger.error(f"Error checking existing server: {e}")

class TestingUI:
    def __init__(self):
        self.logger = logger
        self.initialized = False
        self.sound_manager = None

        try:
            # Create required directories
            self.macro_dir = Path('macros')
            self.macro_dir.mkdir(exist_ok=True)
            self.logger.info(f"Created macros directory at {self.macro_dir}")

            self.screenshots_dir = Path('debug_screenshots')
            self.screenshots_dir.mkdir(exist_ok=True)
            self.logger.info(f"Created screenshots directory at {self.screenshots_dir}")

            # Ensure hotkeys file exists
            self.hotkeys_file = Path('macro_hotkeys.json')
            if not self.hotkeys_file.exists():
                with open(self.hotkeys_file, 'w') as f:
                    json.dump({}, f, indent=2)
                self.logger.info(f"Created hotkeys file at {self.hotkeys_file}")

            # Initialize sound macro manager
            self.sound_manager = SoundMacroManager(test_mode=True)

            # Create test macro if it doesn't exist
            self._create_test_macro()

            self.initialized = True
            self.logger.info("Testing UI initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing TestingUI: {e}")
            self.logger.error(traceback.format_exc())
            raise

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

    def take_screenshot(self, macro_name):
        """Take a screenshot and save it"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{macro_name or 'screenshot'}_{timestamp}.png"
            filepath = self.screenshots_dir / filename
            screenshot = ImageGrab.grab()
            screenshot.save(filepath)
            return True, str(filepath), filename
        except Exception as e:
            self.logger.error(f"Error taking screenshot: {e}")
            return False, None, None

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.sound_manager:
                self.logger.info("Cleaning up sound manager")
            self.logger.info("Resources cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


@app.route('/')
def index():
    """Flask route for web interface"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {e}")
        logger.error(traceback.format_exc())
        return f"Error: {str(e)}", 500

@app.route('/screenshots/<path:filename>')
def serve_screenshot(filename):
    """Serve screenshot files"""
    try:
        return send_file(str(testing_ui.screenshots_dir / filename))
    except Exception as e:
        logger.error(f"Error serving screenshot: {e}")
        return f"Error: {str(e)}", 500

@sock.route('/ws')
def websocket(ws):
    """WebSocket endpoint"""
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
                logger.info(f"Received command: {data['type']}")

                if data['type'] == 'ping':
                    ws.send(json.dumps({'type': 'pong'}))
                    continue

                if data['type'] == 'update_context':
                    handle_update_context(ws, data)
                elif data['type'] == 'get_suggestions':
                    handle_get_suggestions(ws, data)
                elif data['type'] == 'take_screenshot':
                    handle_take_screenshot(ws, data)
                elif data['type'] == 'assign_hotkey':
                    handle_assign_hotkey(ws, data)
                elif data['type'] == 'clear_hotkey':
                    handle_clear_hotkey(ws, data)

            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
                ws.send(json.dumps({
                    'type': 'error',
                    'message': 'Invalid message format'
                }))
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

def handle_update_context(ws, data):
    """Handle macro context update"""
    macro_name = data.get('macro_name', '')
    window_title = data.get('window_title', '')
    active_app = data.get('active_app', '')
    recent_keys = data.get('recent_keys', [])

    if not macro_name:
        ws.send(json.dumps({
            'type': 'error',
            'message': 'Please provide a macro name'
        }))
        return

    try:
        success = testing_ui.sound_manager.update_macro_context(
            macro_name, window_title, active_app, recent_keys
        )

        if success:
            ws.send(json.dumps({
                'type': 'log',
                'level': 'info',
                'message': f'Updated context for macro: {macro_name}'
            }))
        else:
            ws.send(json.dumps({
                'type': 'error',
                'message': 'Failed to update macro context'
            }))
    except Exception as e:
        logger.error(f"Error updating macro context: {e}")
        ws.send(json.dumps({
            'type': 'error',
            'message': f'Error updating context: {str(e)}'
        }))

def handle_get_suggestions(ws, data):
    """Handle macro suggestions request"""
    window_title = data.get('window_title', '')
    active_app = data.get('active_app', '')
    recent_keys = data.get('recent_keys', [])
    max_suggestions = data.get('max_suggestions', 3)

    try:
        suggestions = testing_ui.sound_manager.get_macro_suggestions(
            window_title, active_app, recent_keys, max_suggestions
        )

        ws.send(json.dumps({
            'type': 'suggestions',
            'suggestions': suggestions
        }))
    except Exception as e:
        logger.error(f"Error getting macro suggestions: {e}")
        ws.send(json.dumps({
            'type': 'error',
            'message': f'Error getting suggestions: {str(e)}'
        }))

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

    try:
        logger.info(f"Attempting to assign hotkey '{hotkey}' to macro '{macro_name}'")
        success = testing_ui.sound_manager.assign_hotkey(macro_name, hotkey)

        if success:
            # Send success log
            ws.send(json.dumps({
                'type': 'log',
                'level': 'info',
                'message': f'Assigned hotkey {hotkey} to macro {macro_name}'
            }))

            # Send hotkey update event
            ws.send(json.dumps({
                'type': 'hotkey_updated',
                'macro_name': macro_name,
                'hotkey': hotkey
            }))
        else:
            logger.error(f"Failed to assign hotkey '{hotkey}' to macro '{macro_name}'")
            ws.send(json.dumps({
                'type': 'error',
                'message': 'Failed to assign hotkey'
            }))
    except Exception as e:
        logger.error(f"Error assigning hotkey: {e}")
        logger.error(traceback.format_exc())
        ws.send(json.dumps({
            'type': 'error',
            'message': f'Error assigning hotkey: {str(e)}'
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

    try:
        logger.info(f"Attempting to clear hotkey for macro '{macro_name}'")
        success = testing_ui.sound_manager.remove_hotkey(macro_name)

        if success:
            # Send success log
            ws.send(json.dumps({
                'type': 'log',
                'level': 'info',
                'message': f'Cleared hotkey for macro {macro_name}'
            }))

            # Send hotkey update event
            ws.send(json.dumps({
                'type': 'hotkey_updated',
                'macro_name': macro_name,
                'hotkey': None
            }))
        else:
            ws.send(json.dumps({
                'type': 'error',
                'message': 'No hotkey assigned to clear'
            }))
    except Exception as e:
        logger.error(f"Error clearing hotkey: {e}")
        logger.error(traceback.format_exc())
        ws.send(json.dumps({
            'type': 'error',
            'message': f'Error clearing hotkey: {str(e)}'
        }))

def run():
    """Start the server"""
    try:
        global testing_ui
        logger.info("Initializing Testing UI server...")

        # Clean up any existing server
        cleanup_existing_server()

        # Check if port is available after cleanup
        if not verify_port_available(5000):
            logger.error("Port 5000 is still not available after cleanup")
            raise RuntimeError("Port 5000 is not available")

        # First clean up any existing instance
        if testing_ui:
            testing_ui.cleanup()

        testing_ui = TestingUI()

        if not testing_ui.initialized:
            logger.error("Failed to initialize Testing UI")
            raise RuntimeError("Failed to initialize Testing UI")

        logger.info("Starting Testing UI server on port 5000")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

    except Exception as e:
        logger.error(f"Server error: {e}")
        logger.error(traceback.format_exc())
        if testing_ui:
            testing_ui.cleanup()
        raise

if __name__ == "__main__":
    run()