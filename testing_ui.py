"""Real-time testing interface for the fishing bot systems"""
import logging
import json
import os
import time
import platform
from flask import Flask, render_template, jsonify, request
from flask_sock import Sock

from map_manager import MapManager
from mock_environment import MockEnvironment
from sound_macro_manager import SoundMacroManager

app = Flask(__name__)
sock = Sock(app)

# Global TestingUI instance
testing_ui = None

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
        # Setup logging
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
        self.sound_manager = SoundMacroManager(test_mode=(platform.system() != 'Windows'))

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
                    }
                ],
                "created": time.time()
            }
            with open(test_macro_path, 'w') as f:
                json.dump(test_macro, f, indent=2)
                self.logger.info(f"Created test macro at {test_macro_path}")

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
        'window': 'Not Detected' if not TestingUI.window_detected else TestingUI.last_window_name,
        'bot_status': 'Running' if TestingUI.bot_running else 'Stopped',
        'learning': TestingUI.learning_mode
    })

@app.route('/sounds')
def get_sounds():
    """API endpoint for getting available sound triggers"""
    global testing_ui
    try:
        if not testing_ui:
            return jsonify({'error': 'Server not initialized', 'sounds': []}), 500

        sounds = testing_ui.sound_manager.sound_trigger.get_trigger_names()
        testing_ui.logger.info(f"Found {len(sounds)} sound triggers: {sounds}")

        response = jsonify({'sounds': sounds})
        response.headers['Content-Type'] = 'application/json'
        response.headers['Cache-Control'] = 'no-store'
        return response

    except Exception as e:
        testing_ui.logger.error(f"Error loading sounds: {str(e)}")
        return jsonify({'error': str(e), 'sounds': []}), 500

@app.route('/macros')
def get_macros():
    """API endpoint for getting available macros"""
    global testing_ui
    try:
        if not testing_ui:
            logging.error("TestingUI instance not initialized")
            response = jsonify({
                'error': 'Server not initialized',
                'macros': []
            })
            response.headers['Content-Type'] = 'application/json'
            return response, 500

        if not os.path.exists(testing_ui.macro_dir):
            os.makedirs(testing_ui.macro_dir)
            testing_ui.logger.info(f"Created macros directory")

        # List all .json files in the macros directory
        macro_files = [f[:-5] for f in os.listdir(testing_ui.macro_dir) 
                      if f.endswith('.json')]

        testing_ui.logger.info(f"Found {len(macro_files)} macros: {macro_files}")

        # Create response with explicit headers
        response = jsonify({
            'macros': macro_files
        })
        response.headers['Content-Type'] = 'application/json'
        response.headers['Cache-Control'] = 'no-store'
        testing_ui.logger.info("Returning macros list as JSON")
        return response

    except Exception as e:
        testing_ui.logger.error(f"Error loading macros: {str(e)}")
        response = jsonify({
            'error': str(e),
            'macros': []
        })
        response.headers['Content-Type'] = 'application/json'
        response.headers['Cache-Control'] = 'no-store'
        return response, 500

@sock.route('/updates')
def updates(ws):
    """WebSocket endpoint for real-time updates and commands"""
    global testing_ui
    while True:
        try:
            message = ws.receive()
            data = json.loads(message)
            TestingUI.logger.info(f"Received command: {data['type']}")

            if data['type'] == 'start_sound_recording':
                sound_name = data.get('sound_name', '')
                if not sound_name:
                    TestingUI.logger.error("No sound name provided")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'data': 'Please provide a name for the sound trigger'
                    }))
                    return

                if testing_ui:
                    try:
                        # Only trigger recording, don't save yet
                        if testing_ui.sound_manager.record_sound_trigger(sound_name, duration=2.0, save=False):
                            TestingUI.logger.info(f"Successfully recorded sound: {sound_name}")
                            ws.send(json.dumps({
                                'type': 'log',
                                'level': 'info',
                                'data': f'Successfully recorded sound: {sound_name}'
                            }))
                        else:
                            ws.send(json.dumps({
                                'type': 'log',
                                'level': 'error',
                                'data': f'Failed to record sound: {sound_name}'
                            }))
                    except Exception as e:
                        TestingUI.logger.error(f"Error recording sound: {str(e)}")
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'error',
                            'data': f'Error recording sound: {str(e)}'
                        }))

            elif data['type'] == 'save_sound_recording':
                sound_name = data.get('sound_name', '')
                if not sound_name:
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'data': 'No sound name provided'
                    }))
                    return

                if testing_ui:
                    try:
                        # Save the last recorded sound
                        if testing_ui.sound_manager.save_sound_trigger(sound_name):
                            TestingUI.logger.info(f"Successfully saved sound: {sound_name}")
                            ws.send(json.dumps({
                                'type': 'log',
                                'level': 'info',
                                'data': f'Successfully saved sound trigger: {sound_name}'
                            }))
                            # Refresh sound list
                            sounds = testing_ui.sound_manager.sound_trigger.get_trigger_names()
                            ws.send(json.dumps({
                                'type': 'sounds_updated',
                                'sounds': sounds
                            }))
                        else:
                            ws.send(json.dumps({
                                'type': 'log',
                                'level': 'error',
                                'data': f'Failed to save sound: {sound_name}'
                            }))
                    except Exception as e:
                        TestingUI.logger.error(f"Error saving sound: {str(e)}")
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'error',
                            'data': f'Error saving sound: {str(e)}'
                        }))

            elif data['type'] == 'map_sound_to_macro':
                sound_name = data.get('sound_name')
                macro_name = data.get('macro_name')
                if not sound_name or not macro_name:
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'data': 'Please select both a sound trigger and macro'
                    }))
                    return

                if testing_ui:
                    if testing_ui.sound_manager.assign_macro_to_sound(sound_name, macro_name):
                        TestingUI.logger.info(f"Successfully mapped sound '{sound_name}' to macro '{macro_name}'")
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'info',
                            'data': f'Successfully mapped sound to macro'
                        }))
                    else:
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'error',
                            'data': 'Failed to map sound to macro'
                        }))

            elif data['type'] == 'start_sound_monitoring':
                if testing_ui:
                    if testing_ui.sound_manager.start_monitoring():
                        TestingUI.logger.info("Started sound monitoring")
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'info',
                            'data': 'Started sound monitoring'
                        }))
                    else:
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'error',
                            'data': 'Failed to start sound monitoring'
                        }))

            elif data['type'] == 'stop_sound_monitoring':
                if testing_ui:
                    testing_ui.sound_manager.stop_monitoring()
                    TestingUI.logger.info("Stopped sound monitoring")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'info',
                        'data': 'Stopped sound monitoring'
                    }))

            elif data['type'] == 'save_macro':
                macro_name = data.get('macro_name', '')
                actions = data.get('actions', [])
                if not macro_name:
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'data': 'Please provide a macro name'
                    }))
                    return

                try:
                    macro_path = os.path.join(testing_ui.macro_dir, f"{macro_name}.json")
                    macro_data = {
                        'name': macro_name,
                        'actions': actions,
                        'created': time.time()
                    }

                    with open(macro_path, 'w') as f:
                        json.dump(macro_data, f, indent=2)
                        TestingUI.logger.info(f"Successfully saved macro: {macro_name}")
                        ws.send(json.dumps({
                            'type': 'completed_recording',
                            'recording_type': 'macro',
                            'name': macro_name
                        }))
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'info',
                            'data': f'Saved macro: {macro_name}'
                        }))
                        # Refresh macro list
                        macro_files = [f[:-5] for f in os.listdir(testing_ui.macro_dir) 
                                        if f.endswith('.json')]
                        ws.send(json.dumps({
                            'type': 'macros_updated',
                            'macros': macro_files
                        }))
                except Exception as e:
                    TestingUI.logger.error(f"Error saving macro: {str(e)}")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'data': f'Failed to save macro: {str(e)}'
                    }))

            elif data['type'] == 'play_macro':
                macro_name = data.get('macro_name')
                if not macro_name:
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'data': 'Please select a macro to play'
                    }))
                    return

                try:
                    macro_path = os.path.join(testing_ui.macro_dir, f"{macro_name}.json")
                    if os.path.exists(macro_path):
                        with open(macro_path, 'r') as f:
                            macro_data = json.load(f)
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
                            'data': f'Macro not found: {macro_name}'
                        }))
                except Exception as e:
                    TestingUI.logger.error(f"Error playing macro: {str(e)}")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'data': f'Failed to play macro: {str(e)}'
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
        if testing_ui:
            testing_ui.logger.error(f"Server error: {str(e)}")
    finally:
        if testing_ui and TestingUI.mock_env and TestingUI.mock_env.is_running:
            TestingUI.mock_env.stop_simulation()
        if testing_ui:
            testing_ui.logger.info("Testing UI server stopped")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run server
    run()