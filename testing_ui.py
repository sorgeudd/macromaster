"""Real-time testing interface for the fishing bot systems

This UI provides a comprehensive testing environment for:
- Player position detection on minimap
- Resource location detection
- Terrain type simulation
- Movement and navigation testing

The interface includes:
- Real-time minimap preview
- Position testing controls
- Terrain type selection
- Resource map loading and testing
- Detailed debug information panel
"""

import sys
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from pathlib import Path
import flask
from flask import Flask, render_template, jsonify
from threading import Thread
import json
from flask_sock import Sock

from map_manager import MapManager
from mock_environment import MockEnvironment

app = Flask(__name__)
sock = Sock(app)

class TestingUI:
    def __init__(self):
        # Setup logging with file and console output
        self.logger = logging.getLogger('TestingUI')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        if not self.logger.handlers:
            file_handler = logging.FileHandler('testing_ui.log')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)

        # Initialize root window
        self.root = tk.Tk()
        self.root.title("Fishing Bot Testing Interface")
        self.root.geometry("800x600")

        # Game window detection state
        self.window_detected = False
        self.last_window_name = ""

        # Initialize core components
        self.map_manager = MapManager()  # Handles map and resource detection
        self.mock_env = MockEnvironment()  # Simulates game environment
        self.mock_env.start_simulation()

        # Start Flask server in a separate thread
        self.flask_thread = Thread(target=self._run_flask)
        self.flask_thread.daemon = True
        self.flask_thread.start()

        self.setup_ui()
        self.logger.info("Testing UI initialized successfully")

    def _run_flask(self):
        """Run Flask server for web feedback"""
        app.run(host='0.0.0.0', port=5000)

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

                if data['type'] == 'detect_window':
                    window_name = data['data']
                    # Call the window detection method
                    ws.send(json.dumps({
                        'type': 'log',
                        'data': f'Attempting to detect window: {window_name}'
                    }))
                    TestingUI.window_detected = True
                    TestingUI.last_window_name = window_name

                elif data['type'] == 'move_to':
                    pos = data['data']
                    ws.send(json.dumps({
                        'type': 'log',
                        'data': f'Moving to position: ({pos["x"]}, {pos["y"]})'
                    }))
                    # Send the movement command

                elif data['type'] == 'set_terrain':
                    terrain = data['data']
                    ws.send(json.dumps({
                        'type': 'log',
                        'data': f'Setting terrain type: {terrain}'
                    }))
                    # Set the terrain type

                elif data['type'] == 'load_map':
                    map_type = data['data']
                    ws.send(json.dumps({
                        'type': 'log',
                        'data': f'Loading {map_type} map'
                    }))
                    # Load the specified map

                elif data['type'] == 'find_nearby':
                    ws.send(json.dumps({
                        'type': 'log',
                        'data': 'Finding nearby resources'
                    }))
                    # Search for nearby resources

                # Send updated status
                ws.send(json.dumps({
                    'type': 'status_update',
                    'data': {
                        'position': 'Active',
                        'resource': 'Ready',
                        'terrain': 'Ready',
                        'window': 'Not Detected' if not TestingUI.window_detected else TestingUI.last_window_name
                    }
                }))

            except Exception as e:
                TestingUI.logger.error(f"WebSocket error: {str(e)}")
                break

    def setup_ui(self):
        """Setup all UI components and layout"""
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Setup control panel
        self._setup_control_panel(main_frame)

        # Setup preview area
        self._setup_preview_area(main_frame)

        # Setup info panel
        self._setup_info_panel(main_frame)

        # Status bar at bottom
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # Start preview updates
        self.update_preview()

    def _setup_control_panel(self, parent):
        """Setup the left control panel with all testing controls"""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="5")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Game window detection frame
        window_frame = ttk.LabelFrame(control_frame, text="Game Window", padding="5")
        window_frame.pack(fill=tk.X, padx=5, pady=5)

        # Window name entry
        name_frame = ttk.Frame(window_frame)
        name_frame.pack(fill=tk.X, pady=2)
        ttk.Label(name_frame, text="Window Name:").pack(side=tk.LEFT, padx=2)
        self.window_name_var = tk.StringVar(value="Your Game Window Name")
        ttk.Entry(name_frame, textvariable=self.window_name_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Window status and detection button
        status_frame = ttk.Frame(window_frame)
        status_frame.pack(fill=tk.X, pady=2)
        self.window_status_var = tk.StringVar(value="Status: Not Detected")
        ttk.Label(status_frame, textvariable=self.window_status_var).pack(side=tk.LEFT, padx=2)
        ttk.Button(status_frame, text="Detect Window", command=self.test_window_detection).pack(side=tk.RIGHT, padx=2)

        # Position testing controls
        pos_frame = ttk.LabelFrame(control_frame, text="Position Testing", padding="5")
        pos_frame.pack(fill=tk.X, padx=5, pady=5)

        # Position coordinates entry
        pos_entry_frame = ttk.Frame(pos_frame)
        pos_entry_frame.pack(fill=tk.X, pady=2)
        ttk.Label(pos_entry_frame, text="X:").pack(side=tk.LEFT, padx=2)
        self.x_var = tk.StringVar(value="100")
        ttk.Entry(pos_entry_frame, textvariable=self.x_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(pos_entry_frame, text="Y:").pack(side=tk.LEFT, padx=2)
        self.y_var = tk.StringVar(value="100")
        ttk.Entry(pos_entry_frame, textvariable=self.y_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(pos_entry_frame, text="Move To", command=self.test_move_to).pack(side=tk.RIGHT, padx=2)

        # Terrain testing controls
        terrain_frame = ttk.LabelFrame(control_frame, text="Terrain Testing", padding="5")
        terrain_frame.pack(fill=tk.X, padx=5, pady=5)

        # Terrain type selection
        self.terrain_var = tk.StringVar(value="normal")
        for i, terrain in enumerate(['normal', 'water', 'mountain', 'forest']):
            ttk.Radiobutton(terrain_frame, text=terrain.capitalize(),
                          variable=self.terrain_var, value=terrain,
                          command=self.change_terrain).pack(fill=tk.X, pady=1)

        # Resource detection controls
        resource_frame = ttk.LabelFrame(control_frame, text="Resource Detection", padding="5")
        resource_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(resource_frame, text="Load Fish Map",
                  command=lambda: self.load_resource_map("fish")).pack(fill=tk.X, pady=2)
        ttk.Button(resource_frame, text="Load Ore Map",
                  command=lambda: self.load_resource_map("ore")).pack(fill=tk.X, pady=2)
        ttk.Button(resource_frame, text="Find Nearby",
                  command=self.find_nearby_resources).pack(fill=tk.X, pady=2)

    def test_window_detection(self):
        """Test if the game window can be detected"""
        window_name = self.window_name_var.get()
        # In real implementation, this would check if the window exists
        self.window_detected = True
        self.last_window_name = window_name
        self.window_status_var.set(f"Status: Detected - {window_name}")
        self.log_info(f"Game window detected: {window_name}")

    def _setup_preview_area(self, parent):
        """Setup the minimap preview area"""
        preview_frame = ttk.LabelFrame(parent, text="Minimap Preview", padding="5")
        preview_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.minimap_label = ttk.Label(preview_frame)
        self.minimap_label.pack(expand=True, fill=tk.BOTH)

    def _setup_info_panel(self, parent):
        """Setup the debug information panel"""
        info_frame = ttk.LabelFrame(parent, text="Status & Debug Info", padding="5")
        info_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.info_text = tk.Text(info_frame, height=10, width=50)
        self.info_text.pack(expand=True, fill=tk.BOTH)

    def update_preview(self):
        """Update minimap preview with current state"""
        try:
            # Create base minimap
            minimap = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.rectangle(minimap, (0, 0), (200, 200), (20, 20, 20), -1)

            # Draw player position if available
            if self.mock_env.state.current_position:
                x, y = self.mock_env.state.current_position
                cv2.circle(minimap, (int(x), int(y)), 3, (0, 255, 0), -1)

            # Show nearby resources if available
            if hasattr(self, 'nearby_resources'):
                for resource in self.nearby_resources:
                    rx, ry = resource.position
                    cv2.circle(minimap, (int(rx), int(ry)), 5, (255, 0, 0), 1)

            # Convert to Tkinter image
            image = Image.fromarray(cv2.cvtColor(minimap, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image)

            self.minimap_label.configure(image=photo)
            self.minimap_label.image = photo  # Keep reference

        except Exception as e:
            self.logger.error(f"Error updating preview: {str(e)}")

        # Schedule next update
        self.root.after(100, self.update_preview)

    def log_info(self, message):
        """Add message to info text panel and broadcast to web clients"""
        self.info_text.insert(tk.END, f"{message}\n")
        self.info_text.see(tk.END)

    def test_move_to(self):
        """Test movement to specified coordinates"""
        try:
            if not self.window_detected:
                messagebox.showwarning("Warning", "Game window not detected!")
                return

            x = int(self.x_var.get())
            y = int(self.y_var.get())

            self.status_var.set(f"Moving to ({x}, {y})")
            success = self.mock_env.move_to((x, y))

            if success:
                self.log_info(f"Successfully moved to ({x}, {y})")
            else:
                self.log_info(f"Failed to move to ({x}, {y})")

        except ValueError:
            messagebox.showerror("Error", "Invalid coordinates")

    def change_terrain(self):
        """Update current terrain type and movement effects"""
        if not self.window_detected:
            messagebox.showwarning("Warning", "Game window not detected!")
            return

        terrain = self.terrain_var.get()
        success = self.mock_env.set_game_state(terrain_type=terrain)

        if success:
            self.log_info(f"Changed terrain to: {terrain}")
            self.status_var.set(f"Current terrain: {terrain}")
        else:
            self.log_info(f"Failed to change terrain to: {terrain}")

    def load_resource_map(self, resource_type):
        """Load and analyze resource map"""
        if not self.window_detected:
            messagebox.showwarning("Warning", "Game window not detected!")
            return

        try:
            map_name = f"mase knoll {resource_type}"
            success = self.map_manager.load_map(map_name)

            if success:
                self.log_info(f"Loaded {resource_type} map")
                self.status_var.set(f"Current map: {map_name}")
            else:
                self.log_info(f"Failed to load {resource_type} map")

        except Exception as e:
            self.logger.error(f"Error loading map: {str(e)}")
            messagebox.showerror("Error", f"Failed to load map: {str(e)}")

    def find_nearby_resources(self):
        """Find resources near current position"""
        if not self.window_detected:
            messagebox.showwarning("Warning", "Game window not detected!")
            return

        try:
            if not self.mock_env.state.current_position:
                messagebox.showwarning("Warning", "No current position set")
                return

            pos = self.mock_env.state.current_position
            self.nearby_resources = self.map_manager.get_nearby_resources(pos, 50)

            self.log_info(f"Found {len(self.nearby_resources)} nearby resources")
            self.status_var.set(f"Found {len(self.nearby_resources)} resources")

        except Exception as e:
            self.logger.error(f"Error finding resources: {str(e)}")
            messagebox.showerror("Error", f"Failed to find resources: {str(e)}")

    def run(self):
        """Start the UI and handle cleanup"""
        try:
            self.root.mainloop()
        finally:
            self.mock_env.stop_simulation()

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)

    # Create and run UI
    ui = TestingUI()
    ui.run()