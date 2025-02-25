"""Real-time testing interface for the fishing bot systems"""
import logging
import json
import os
import platform
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
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'data': 'Please enter a sound trigger name'
                    }))
                    return

                try:
                    if testing_ui.sound_manager.record_sound_trigger(sound_name, duration=2.0, save=False):
                        TestingUI.logger.info(f"Successfully started recording sound: {sound_name}")
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'info',
                            'data': f'Recording started: {sound_name}'
                        }))
                    else:
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'error',
                            'data': f'Failed to start recording'
                        }))
                except Exception as e:
                    TestingUI.logger.error(f"Error recording sound: {str(e)}")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'data': f'Error recording sound: {str(e)}'
                    }))

            elif data['type'] == 'stop_sound_recording':
                try:
                    if testing_ui.sound_manager.stop_recording():
                        TestingUI.logger.info("Successfully stopped sound recording")
                        ws.send(json.dumps({
                            'type': 'recording_complete',
                            'recording_type': 'sound'
                        }))
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'info',
                            'data': 'Recording stopped successfully'
                        }))
                    else:
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'error',
                            'data': 'Failed to stop recording'
                        }))
                except Exception as e:
                    TestingUI.logger.error(f"Error stopping recording: {str(e)}")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'data': f'Error stopping recording: {str(e)}'
                    }))

            elif data['type'] == 'save_sound_recording':
                sound_name = data.get('sound_name', '')
                if not sound_name:
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'data': 'Please provide a sound name'
                    }))
                    return

                try:
                    if testing_ui.sound_manager.save_sound_trigger(sound_name):
                        TestingUI.logger.info(f"Successfully saved sound: {sound_name}")
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'info',
                            'data': f'Sound saved successfully'
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
                            'data': f'Failed to save sound'
                        }))
                except Exception as e:
                    TestingUI.logger.error(f"Error saving sound: {str(e)}")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'data': f'Error saving sound: {str(e)}'
                    }))

            elif data['type'] == 'start_macro_recording':
                macro_name = data.get('macro_name', '')
                if not macro_name:
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'data': 'Please provide a macro name'
                    }))
                    return

                try:
                    TestingUI.recording_macro = True
                    TestingUI.logger.info(f"Started recording macro: {macro_name}")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'info',
                        'data': f'Started recording macro: {macro_name}'
                    }))
                except Exception as e:
                    TestingUI.logger.error(f"Error starting macro recording: {str(e)}")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'data': f'Error starting macro recording: {str(e)}'
                    }))

            elif data['type'] == 'stop_macro_recording':
                try:
                    if testing_ui.sound_manager.stop_macro_recording():
                        TestingUI.logger.info("Successfully stopped macro recording")
                        ws.send(json.dumps({
                            'type': 'recording_complete',
                            'recording_type': 'macro'
                        }))
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'info',
                            'data': 'Macro recording stopped successfully'
                        }))
                    else:
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'error',
                            'data': 'Failed to stop macro recording'
                        }))
                except Exception as e:
                    TestingUI.logger.error(f"Error stopping macro recording: {str(e)}")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'data': f'Error stopping macro recording: {str(e)}'
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
                    import time
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
                        'type': 'log',
                        'level': 'info',
                        'data': f'Macro saved successfully'
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
                        'data': f'Error saving macro: {str(e)}'
                    }))

            elif data['type'] == 'map_sound_to_macro':
                sound_name = data.get('data', {}).get('sound_name')
                macro_name = data.get('data', {}).get('macro_name')
                if not sound_name or not macro_name:
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'data': 'Please select both a sound trigger and macro'
                    }))
                    return

                try:
                    if testing_ui.sound_manager.assign_macro_to_sound(sound_name, macro_name):
                        TestingUI.logger.info(f"Mapped sound '{sound_name}' to macro '{macro_name}'")
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'info',
                            'data': 'Sound trigger mapped to macro successfully'
                        }))
                    else:
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'error',
                            'data': 'Failed to map sound trigger to macro'
                        }))
                except Exception as e:
                    TestingUI.logger.error(f"Error mapping sound to macro: {str(e)}")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'data': f'Error mapping sound to macro: {str(e)}'
                    }))

            elif data['type'] in ['start_sound_monitoring', 'stop_sound_monitoring']:
                try:
                    if data['type'] == 'start_sound_monitoring':
                        if testing_ui.sound_manager.start_monitoring():
                            TestingUI.logger.info("Started sound monitoring")
                            ws.send(json.dumps({
                                'type': 'log',
                                'level': 'info',
                                'data': 'Sound monitoring started'
                            }))
                    else:
                        testing_ui.sound_manager.stop_monitoring()
                        TestingUI.logger.info("Stopped sound monitoring")
                        ws.send(json.dumps({
                            'type': 'log',
                            'level': 'info',
                            'data': 'Sound monitoring stopped'
                        }))
                except Exception as e:
                    TestingUI.logger.error(f"Error with sound monitoring: {str(e)}")
                    ws.send(json.dumps({
                        'type': 'log',
                        'level': 'error',
                        'data': f'Error with sound monitoring: {str(e)}'
                    }))

        except Exception as e:
            TestingUI.logger.error(f"Error processing command: {str(e)}")
            ws.send(json.dumps({
                'type': 'log',
                'level': 'error',
                'data': f'Error processing command: {str(e)}'
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