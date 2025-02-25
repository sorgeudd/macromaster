"""Real-time testing interface for the fishing bot systems"""

import logging
import json
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
        self.logger.setLevel(logging.DEBUG)

        # Initialize components
        self.map_manager = MapManager()
        self.mock_env = MockEnvironment()
        self.mock_env.start_simulation()

        self.logger.info("Testing UI initialized successfully")

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
            'window': 'Not Detected' if not TestingUI.window_detected else TestingUI.last_window_name
        })

    @sock.route('/updates')
    def updates(ws):
        """WebSocket endpoint for real-time updates and commands"""
        while True:
            try:
                message = ws.receive()
                data = json.loads(message)

                # Send log message for each command
                ws.send(json.dumps({
                    'type': 'log',
                    'level': 'info',
                    'data': f"Received command: {data['type']}"
                }))

                if data['type'] == 'detect_window':
                    window_name = data['data']
                    TestingUI.window_detected = True
                    TestingUI.last_window_name = window_name
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'info',
                        'data': f'Game window detected: {window_name}'
                    }))

                elif data['type'] == 'start_bot':
                    TestingUI.bot_running = True
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'info',
                        'data': 'Bot started'
                    }))

                elif data['type'] == 'stop_bot':
                    TestingUI.bot_running = False
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'info',
                        'data': 'Bot stopped'
                    }))

                elif data['type'] == 'emergency_stop':
                    TestingUI.bot_running = False
                    TestingUI.learning_mode = False
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'warning',
                        'data': 'EMERGENCY STOP activated!'
                    }))

                elif data['type'] == 'start_learning':
                    TestingUI.learning_mode = True
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'info',
                        'data': 'Learning mode started'
                    }))

                elif data['type'] == 'reset_learning':
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'info',
                        'data': 'Learning data reset'
                    }))

                elif data['type'] == 'test_arrow_detection':
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'info',
                        'data': 'Running arrow detection test...'
                    }))
                    # Add actual test implementation here

                elif data['type'] == 'test_resource_spots':
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'info',
                        'data': 'Testing resource spot detection...'
                    }))
                    # Add actual test implementation here

                elif data['type'] == 'calibrate_terrain':
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'info',
                        'data': 'Starting terrain calibration...'
                    }))
                    # Add actual calibration implementation here

                elif data['type'] == 'test_pathfinding':
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'info',
                        'data': 'Testing pathfinding system...'
                    }))
                    # Add actual test implementation here

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
            self.logger.info("Starting Testing UI server")
            app.run(host='0.0.0.0', port=5000)
        except Exception as e:
            self.logger.error(f"Server error: {str(e)}")
        finally:
            self.mock_env.stop_simulation()
            self.logger.info("Testing UI server stopped")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)

    # Create and run UI
    ui = TestingUI()
    ui.run()