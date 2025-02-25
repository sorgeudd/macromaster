"""Real-time testing interface for the fishing bot systems"""
import logging
import json
import os
import platform
import time
from flask import Flask, render_template, jsonify
from flask_sock import Sock

from map_manager import MapManager
from mock_environment import MockEnvironment
from sound_macro_manager import SoundMacroManager

app = Flask(__name__)
sock = Sock(app)

# Global TestingUI instance
testing_ui = None

class TestingUI:
    def __init__(self):
        # Setup logging
        self.logger = logging.getLogger('TestingUI')
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
        self.mock_env = MockEnvironment()
        self.sound_manager = SoundMacroManager(test_mode=(platform.system() != 'Windows'))

        # Ensure macros directory exists
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
                        "timestamp": 0
                    }
                ],
                "created": 0
            }
            with open(test_macro_path, 'w') as f:
                json.dump(test_macro, f, indent=2)
                self.logger.info(f"Created test macro at {test_macro_path}")

        self.logger.info("Testing UI initialized successfully")

@app.route('/')
def index():
    """Flask route for web interface"""
    return render_template('index.html')

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
            return jsonify({'error': 'Server not initialized', 'macros': []}), 500

        if not os.path.exists(testing_ui.macro_dir):
            os.makedirs(testing_ui.macro_dir)
            testing_ui.logger.info(f"Created macros directory")

        macro_files = [f[:-5] for f in os.listdir(testing_ui.macro_dir) 
                      if f.endswith('.json')]

        testing_ui.logger.info(f"Found {len(macro_files)} macros: {macro_files}")
        response = jsonify({'macros': macro_files})
        response.headers['Content-Type'] = 'application/json'
        response.headers['Cache-Control'] = 'no-store'
        return response

    except Exception as e:
        testing_ui.logger.error(f"Error loading macros: {str(e)}")
        return jsonify({'error': str(e), 'macros': []}), 500

@sock.route('/ws')  # Changed from '/updates' to '/ws' to match client
def websocket(ws):
    """WebSocket endpoint for real-time updates and commands"""
    global testing_ui
    while True:
        try:
            message = ws.receive()
            data = json.loads(message)
            testing_ui.logger.info(f"Received command: {data['type']}")

            # Handle the different message types
            if data['type'] == 'start_sound_recording':
                sound_name = data.get('sound_name', '')
                if not sound_name:
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'message': 'Please enter a sound trigger name'
                    }))
                    continue

                try:
                    if testing_ui.sound_manager.record_sound_trigger(sound_name, duration=2.0, save=False):
                        testing_ui.logger.info(f"Successfully started recording sound: {sound_name}")
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'info',
                            'message': f'Recording started: {sound_name}'
                        }))
                    else:
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'error',
                            'message': f'Failed to start recording'
                        }))
                except Exception as e:
                    testing_ui.logger.error(f"Error recording sound: {str(e)}")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'message': f'Error recording sound: {str(e)}'
                    }))

            elif data['type'] == 'stop_sound_recording':
                try:
                    if testing_ui.sound_manager.stop_recording():
                        testing_ui.logger.info("Successfully stopped sound recording")
                        ws.send(json.dumps({
                            'type': 'recording_complete',
                            'recording_type': 'sound'
                        }))
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'info',
                            'message': 'Recording stopped successfully'
                        }))
                    else:
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'error',
                            'message': 'Failed to stop recording'
                        }))
                except Exception as e:
                    testing_ui.logger.error(f"Error stopping recording: {str(e)}")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'message': f'Error stopping recording: {str(e)}'
                    }))
            elif data['type'] == 'save_sound_recording':
                sound_name = data.get('sound_name', '')
                if not sound_name:
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'message': 'Please provide a sound name'
                    }))
                    return

                try:
                    if testing_ui.sound_manager.save_sound_trigger(sound_name):
                        testing_ui.logger.info(f"Successfully saved sound: {sound_name}")
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'info',
                            'message': f'Sound saved successfully'
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
                            'message': f'Failed to save sound'
                        }))
                except Exception as e:
                    testing_ui.logger.error(f"Error saving sound: {str(e)}")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'message': f'Error saving sound: {str(e)}'
                    }))

            elif data['type'] == 'start_macro_recording':
                macro_name = data.get('macro_name', '')
                if not macro_name:
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'message': 'Please provide a macro name'
                    }))
                    return

                try:
                    if testing_ui.sound_manager.record_macro(macro_name, duration=5.0):
                        testing_ui.logger.info(f"Started recording macro: {macro_name}")
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'info',
                            'message': f'Started recording macro: {macro_name}'
                        }))
                    else:
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'error',
                            'message': 'Failed to start macro recording'
                        }))
                except Exception as e:
                    testing_ui.logger.error(f"Error starting macro recording: {str(e)}")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'message': f'Error starting macro recording: {str(e)}'
                    }))

            elif data['type'] == 'stop_macro_recording':
                try:
                    if testing_ui.sound_manager.stop_macro_recording():
                        testing_ui.logger.info("Successfully stopped macro recording")
                        ws.send(json.dumps({
                            'type': 'recording_complete',
                            'recording_type': 'macro'
                        }))
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'info',
                            'message': 'Macro recording stopped successfully'
                        }))
                    else:
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'error',
                            'message': 'Failed to stop macro recording'
                        }))
                except Exception as e:
                    testing_ui.logger.error(f"Error stopping macro recording: {str(e)}")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'message': f'Error stopping macro recording: {str(e)}'
                    }))

            elif data['type'] == 'save_macro':
                macro_name = data.get('macro_name', '')
                if not macro_name:
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'message': 'Please provide a macro name'
                    }))
                    return

                try:
                    macro_path = os.path.join(testing_ui.macro_dir, f"{macro_name}.json")
                    macro_data = {
                        'name': macro_name,
                        'actions': [],
                        'created': time.time()
                    }

                    with open(macro_path, 'w') as f:
                        json.dump(macro_data, f, indent=2)
                    testing_ui.logger.info(f"Successfully saved macro: {macro_name}")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'info',
                        'message': f'Macro saved successfully'
                    }))
                    # Refresh macro list
                    macro_files = [f[:-5] for f in os.listdir(testing_ui.macro_dir) 
                                 if f.endswith('.json')]
                    ws.send(json.dumps({
                        'type': 'macros_updated',
                        'macros': macro_files
                    }))
                except Exception as e:
                    testing_ui.logger.error(f"Error saving macro: {str(e)}")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'message': f'Error saving macro: {str(e)}'
                    }))

            elif data['type'] == 'map_sound_to_macro':
                sound_name = data.get('data', {}).get('sound_name')
                macro_name = data.get('data', {}).get('macro_name')
                if not sound_name or not macro_name:
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'message': 'Please select both a sound trigger and macro'
                    }))
                    return

                try:
                    if testing_ui.sound_manager.assign_macro_to_sound(sound_name, macro_name):
                        testing_ui.logger.info(f"Mapped sound '{sound_name}' to macro '{macro_name}'")
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'info',
                            'message': 'Sound trigger mapped to macro successfully'
                        }))
                    else:
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'error',
                            'message': 'Failed to map sound trigger to macro'
                        }))
                except Exception as e:
                    testing_ui.logger.error(f"Error mapping sound to macro: {str(e)}")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'message': f'Error mapping sound to macro: {str(e)}'
                    }))

            elif data['type'] in ['start_sound_monitoring', 'stop_sound_monitoring']:
                try:
                    if data['type'] == 'start_sound_monitoring':
                        if testing_ui.sound_manager.start_monitoring():
                            testing_ui.logger.info("Started sound monitoring")
                            ws.send(json.dumps({
                                'type': 'log',
                                'level': 'info',
                                'message': 'Sound monitoring started'
                            }))
                    else:
                        testing_ui.sound_manager.stop_monitoring()
                        testing_ui.logger.info("Stopped sound monitoring")
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'info',
                            'message': 'Sound monitoring stopped'
                        }))
                except Exception as e:
                    testing_ui.logger.error(f"Error with sound monitoring: {str(e)}")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'message': f'Error with sound monitoring: {str(e)}'
                    }))

            # Bot and learning mode commands
            elif data['type'] == 'start_bot':
                try:
                    testing_ui.logger.info("Starting bot")
                    ws.send(json.dumps({
                        'type': 'status_update',
                        'bot_status': 'Running',
                        'window_status': 'Active',
                        'learning_status': 'Inactive'
                    }))
                except Exception as e:
                    testing_ui.logger.error(f"Error starting bot: {str(e)}")

            elif data['type'] == 'stop_bot':
                try:
                    testing_ui.logger.info("Stopping bot")
                    ws.send(json.dumps({
                        'type': 'status_update',
                        'bot_status': 'Stopped',
                        'window_status': 'Active',
                        'learning_status': 'Inactive'
                    }))
                except Exception as e:
                    testing_ui.logger.error(f"Error stopping bot: {str(e)}")

            elif data['type'] == 'start_learning':
                try:
                    testing_ui.logger.info("Starting learning mode")
                    ws.send(json.dumps({
                        'type': 'status_update',
                        'learning_status': 'Active',
                        'bot_status': 'Stopped',
                        'window_status': 'Active'
                    }))
                except Exception as e:
                    testing_ui.logger.error(f"Error starting learning mode: {str(e)}")

            elif data['type'] == 'stop_learning':
                try:
                    testing_ui.logger.info("Stopping learning mode")
                    ws.send(json.dumps({
                        'type': 'status_update',
                        'learning_status': 'Inactive',
                        'bot_status': 'Stopped',
                        'window_status': 'Active'
                    }))
                except Exception as e:
                    testing_ui.logger.error(f"Error stopping learning mode: {str(e)}")

        except Exception as e:
            testing_ui.logger.error(f"Error processing command: {str(e)}")
            ws.send(json.dumps({
                'type': 'log',
                'level': 'error',
                'message': f'Error processing command: {str(e)}'
            }))

def run():
    """Start the server"""
    global testing_ui
    try:
        testing_ui = TestingUI()
        testing_ui.logger.info("Starting Testing UI server")
        app.run(host='0.0.0.0', port=5000)
    except Exception as e:
        if testing_ui:
            testing_ui.logger.error(f"Server error: {str(e)}")
    finally:
        if testing_ui and testing_ui.mock_env and testing_ui.mock_env.is_running:
            testing_ui.mock_env.stop_simulation()
        if testing_ui:
            testing_ui.logger.info("Testing UI server stopped")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()