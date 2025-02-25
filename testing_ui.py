"""Real-time testing interface for the fishing bot systems"""

import logging
import json
import os
import platform
from flask import Flask, render_template, jsonify
from flask_sock import Sock

from map_manager import MapManager
from mock_environment import MockEnvironment

app = Flask(__name__)
sock = Sock(app)

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

    @staticmethod
    def find_window(window_name):
        """Find window by name with cross-platform support"""
        try:
            # Only import win32gui on Windows
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
                # Mock window detection for testing
                TestingUI.logger.warning("Running in test mode (non-Windows platform)")
                # Simulate window detection for specific names (for testing)
                test_windows = ['albion online client', 'game window']
                found = any(window_name.lower() in test_win for test_win in test_windows)
                return found, 1 if found else None
        except Exception as e:
            TestingUI.logger.error(f"Error detecting window: {str(e)}")
            return False, None

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
        self.logger.setLevel(logging.INFO)  # Changed from DEBUG to INFO to reduce spam

        # Initialize components
        self.map_manager = MapManager()
        # Don't start mock environment automatically
        TestingUI.mock_env = MockEnvironment()

        self.logger.info("Testing UI initialized successfully")
        self.logger.info("Running as local server with full system access")

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
        try:
            macro_dir = 'macros'
            if not os.path.exists(macro_dir):
                os.makedirs(macro_dir)

            macro_files = [f[:-5] for f in os.listdir(macro_dir) 
                         if f.endswith('.json')]

            return jsonify({
                'macros': macro_files
            })
        except Exception as e:
            TestingUI.logger.error(f"Error loading macros: {str(e)}")
            return jsonify({
                'macros': []
            })

    @sock.route('/updates')
    def updates(ws):
        """WebSocket endpoint for real-time updates and commands"""
        while True:
            try:
                message = ws.receive()
                data = json.loads(message)

                # Log received command
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
                    # Only start mock environment when bot starts
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
                    # Stop mock environment when bot stops
                    if TestingUI.mock_env.is_running:
                        TestingUI.mock_env.stop_simulation()
                    TestingUI.logger.info("Stopping bot operations")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'info',
                        'data': 'Bot stopped'
                    }))

                elif data['type'] == 'emergency_stop':
                    TestingUI.bot_running = False
                    TestingUI.learning_mode = False
                    # Stop mock environment on emergency stop
                    if TestingUI.mock_env.is_running:
                        TestingUI.mock_env.stop_simulation()
                    TestingUI.logger.warning("Emergency stop triggered")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'warning',
                        'data': 'EMERGENCY STOP activated!'
                    }))

                elif data['type'] == 'start_learning':
                    TestingUI.learning_mode = True
                    TestingUI.logger.info("Starting learning mode")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'info',
                        'data': 'Learning mode started'
                    }))

                elif data['type'] == 'reset_learning':
                    TestingUI.logger.info("Resetting learning data")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'info',
                        'data': 'Learning data reset'
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

    def run(self):
        """Start the server"""
        try:
            self.logger.info("Starting Testing UI server with full system access")
            app.run(host='0.0.0.0', port=5000)
        except Exception as e:
            self.logger.error(f"Server error: {str(e)}")
        finally:
            if TestingUI.mock_env and TestingUI.mock_env.is_running:
                TestingUI.mock_env.stop_simulation()
            self.logger.info("Testing UI server stopped")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)  # Changed from DEBUG to INFO

    # Create and run UI
    ui = TestingUI()
    ui.run()