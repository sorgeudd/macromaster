"""Real-time testing interface for the fishing bot systems"""

import logging
import json
import os
import platform
import time
from flask import Flask, render_template, jsonify, request
from flask_sock import Sock

from map_manager import MapManager
from mock_environment import MockEnvironment

app = Flask(__name__)
sock = Sock(app)

# Global TestingUI instance
testing_ui = None

@app.route('/')
def index():
    """Flask route for web interface"""
    return render_template('index.html')

@app.route('/status')
def get_status():
    """API endpoint for getting current status"""
    return jsonify({
        'position': 'Active',
        'resource': 'Ready',
        'terrain': 'Ready',
        'window': 'Not Detected' if not TestingUI.window_detected else TestingUI.last_window_name,
        'bot_status': 'Running' if TestingUI.bot_running else 'Stopped',
        'learning': TestingUI.learning_mode
    })

@app.route('/macros')
def get_macros():
    """API endpoint for getting available macros"""
    global testing_ui
    try:
        if not os.path.exists(testing_ui.macro_dir):
            os.makedirs(testing_ui.macro_dir)
            testing_ui.logger.info(f"Created macros directory")

        macro_files = [f[:-5] for f in os.listdir(testing_ui.macro_dir) 
                      if f.endswith('.json')]

        testing_ui.logger.info(f"Found {len(macro_files)} macros: {macro_files}")
        response = jsonify({
            'macros': macro_files
        })
        response.headers['Content-Type'] = 'application/json'
        testing_ui.logger.info("Returning macros list as JSON")
        return response
    except Exception as e:
        testing_ui.logger.error(f"Error loading macros: {str(e)}")
        response = jsonify({
            'error': str(e),
            'macros': []
        })
        response.headers['Content-Type'] = 'application/json'
        return response, 500  # Return 500 status code for server errors

@app.route('/macro/<name>', methods=['GET'])
def get_macro(name):
    """API endpoint for getting a specific macro"""
    global testing_ui
    try:
        if not testing_ui:
            return jsonify({'error': 'Server not initialized'}), 500

        macro_data = testing_ui.load_macro(name)
        if macro_data:
            response = jsonify(macro_data)
            response.headers['Content-Type'] = 'application/json'
            return response
        return jsonify({'error': 'Macro not found'}), 404
    except Exception as e:
        testing_ui.logger.error(f"Error getting macro {name}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/macro/<name>', methods=['POST'])
def save_macro_endpoint(name):
    """API endpoint for saving a macro"""
    global testing_ui
    try:
        if not testing_ui:
            return jsonify({'error': 'Server not initialized'}), 500

        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        actions = request.json.get('actions', [])
        if testing_ui.save_macro(name, actions):
            testing_ui.logger.info(f"Successfully saved macro: {name}")
            return jsonify({'success': True})
        return jsonify({'error': 'Failed to save macro'}), 500
    except Exception as e:
        testing_ui.logger.error(f"Error saving macro {name}: {str(e)}")
        return jsonify({'error': str(e)}), 500

class TestingUI:
    # Class variables for state management
    window_detected = False
    last_window_name = ""
    bot_running = False
    learning_mode = False
    recording_macro = False
    recording_sound = False
    logger = logging.getLogger('TestingUI')
    mock_env = None

    def __init__(self):
        # Setup logging with file and console output
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        if not self.logger.handlers:
            file_handler = logging.FileHandler('testing_ui.log')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)

        # Initialize components
        self.map_manager = MapManager()
        TestingUI.mock_env = MockEnvironment()

        # Ensure macros directory exists and initialize test macro
        self.macro_dir = 'macros'
        if not os.path.exists(self.macro_dir):
            os.makedirs(self.macro_dir)
            self.logger.info(f"Created macros directory at {self.macro_dir}")

        # Create test macro if it doesn't exist
        test_macro_path = os.path.join(self.macro_dir, 'test_macro.json')
        if not os.path.exists(test_macro_path):
            test_macro = {
                "name": "test_macro",
                "actions": [
                    {
                        "type": "move",
                        "x": 100,
                        "y": 100,
                        "button": "left",
                        "duration": 0.1,
                        "timestamp": time.time()
                    },
                    {
                        "type": "click",
                        "x": 100,
                        "y": 100,
                        "button": "left",
                        "duration": 0.1,
                        "timestamp": time.time() + 0.1
                    },
                    {
                        "type": "key_press",
                        "key": "e",
                        "duration": 0.5,
                        "timestamp": time.time() + 0.2
                    }
                ],
                "created": time.time()
            }
            with open(test_macro_path, 'w') as f:
                json.dump(test_macro, f, indent=2)
                self.logger.info(f"Created test macro at {test_macro_path}")

        self.logger.info("Testing UI initialized successfully")

    @staticmethod
    def find_window(window_name):
        """Find window by name with cross-platform support"""
        try:
            if platform.system() == 'Windows':
                import win32gui
                def callback(hwnd, hwnds):
                    if win32gui.IsWindowVisible(hwnd):
                        title = win32gui.GetWindowText(hwnd)
                        if window_name.lower() in title.lower():
                            hwnds.append(hwnd)
                    return True

                hwnds = []
                win32gui.EnumWindows(callback, hwnds)
                return len(hwnds) > 0, hwnds[0] if hwnds else None
            else:
                TestingUI.logger.warning("Running in test mode (non-Windows platform)")
                test_windows = ['albion online client', 'game window']
                found = any(window_name.lower() in test_win for test_win in test_windows)
                return found, 1 if found else None
        except Exception as e:
            TestingUI.logger.error(f"Error detecting window: {str(e)}")
            return False, None

    def load_macro(self, macro_name):
        """Load a macro from file"""
        try:
            macro_path = os.path.join(self.macro_dir, f"{macro_name}.json")
            if not os.path.exists(macro_path):
                self.logger.error(f"Macro file not found: {macro_path}")
                return None

            with open(macro_path, 'r') as f:
                macro_data = json.load(f)
                self.logger.info(f"Successfully loaded macro: {macro_name}")
                return macro_data

        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in macro file: {macro_name}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading macro {macro_name}: {str(e)}")
            return None

    def save_macro(self, macro_name, actions):
        """Save a macro to file"""
        try:
            macro_path = os.path.join(self.macro_dir, f"{macro_name}.json")
            macro_data = {
                'name': macro_name,
                'actions': actions,
                'created': time.time()
            }

            with open(macro_path, 'w') as f:
                json.dump(macro_data, f, indent=2)
                self.logger.info(f"Successfully saved macro: {macro_name}")
                return True

        except Exception as e:
            self.logger.error(f"Error saving macro {macro_name}: {str(e)}")
            return False

@sock.route('/updates')
def updates(ws):
    """WebSocket endpoint for real-time updates and commands"""
    global testing_ui
    while True:
        try:
            message = ws.receive()
            data = json.loads(message)
            TestingUI.logger.info(f"Received command: {data['type']}")

            if data['type'] == 'detect_window':
                window_name = data['data']
                TestingUI.logger.info(f"Attempting to detect window: '{window_name}'")

                found, hwnd = TestingUI.find_window(window_name)
                if found:
                    TestingUI.window_detected = True
                    TestingUI.last_window_name = window_name
                    TestingUI.logger.info(f"Successfully detected window '{window_name}' (hwnd: {hwnd})")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'info',
                        'data': f'Game window detected: {window_name}'
                    }))
                else:
                    TestingUI.window_detected = False
                    TestingUI.last_window_name = ""
                    TestingUI.logger.warning(f"Failed to detect window '{window_name}'")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'data': f'Could not find game window: {window_name}'
                    }))

            elif data['type'] == 'start_bot':
                if not TestingUI.mock_env.is_running:
                    TestingUI.mock_env.start_simulation()
                TestingUI.bot_running = True
                TestingUI.logger.info("Starting bot operations")
                ws.send(json.dumps({
                    'type': 'log',
                    'level': 'info',
                    'data': 'Bot started'
                }))

            elif data['type'] == 'stop_bot':
                TestingUI.bot_running = False
                if TestingUI.mock_env.is_running:
                    TestingUI.mock_env.stop_simulation()
                TestingUI.logger.info("Stopping bot operations")
                ws.send(json.dumps({
                    'type': 'log',
                    'level': 'info',
                    'data': 'Bot stopped'
                }))

            elif data['type'] == 'start_macro_recording':
                TestingUI.recording_macro = True
                TestingUI.logger.info("Started macro recording")
                ws.send(json.dumps({
                    'type': 'log',
                    'level': 'info',
                    'data': 'Started recording macro'
                }))

            elif data['type'] == 'stop_macro_recording':
                macro_name = data.get('macro_name', 'unnamed_macro')
                actions = data.get('actions', [])
                if testing_ui.save_macro(macro_name, actions):
                    TestingUI.logger.info(f"Saved macro: {macro_name}")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'info',
                        'data': f'Saved macro: {macro_name}'
                    }))
                else:
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'data': f'Failed to save macro: {macro_name}'
                    }))
                TestingUI.recording_macro = False

            elif data['type'] == 'play_macro':
                macro_name = data.get('macro_name')
                if macro_name:
                    macro_data = testing_ui.load_macro(macro_name)
                    if macro_data:
                        TestingUI.logger.info(f"Playing macro: {macro_name}")
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'info',
                            'data': f'Playing macro: {macro_name}'
                        }))
                    else:
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'error',
                            'data': f'Failed to load macro: {macro_name}'
                        }))

            # Send updated status after each command
            ws.send(json.dumps({
                'type': 'status_update',
                'data': {
                    'position': 'Active',
                    'resource': 'Ready',
                    'terrain': 'Ready',
                    'window': 'Not Detected' if not TestingUI.window_detected else TestingUI.last_window_name,
                    'bot_status': 'Running' if TestingUI.bot_running else 'Stopped',
                    'learning': TestingUI.learning_mode,
                    'recording': TestingUI.recording_macro or TestingUI.recording_sound
                }
            }))

        except Exception as e:
            error_msg = f"Error processing command: {str(e)}"
            TestingUI.logger.error(error_msg)
            ws.send(json.dumps({
                'type': 'log',
                'level': 'error',
                'data': error_msg
            }))

def run():
    """Start the server"""
    global testing_ui
    try:
        # Create TestingUI instance
        testing_ui = TestingUI()
        testing_ui.logger.info("Starting Testing UI server with full system access")
        app.run(host='0.0.0.0', port=5000)
    except Exception as e:
        testing_ui.logger.error(f"Server error: {str(e)}")
    finally:
        if TestingUI.mock_env and TestingUI.mock_env.is_running:
            TestingUI.mock_env.stop_simulation()
        testing_ui.logger.info("Testing UI server stopped")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run server
    run()