"""Real-time testing interface for the fishing bot systems"""
import sys
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from pathlib import Path

from map_manager import MapManager
from mock_environment import MockEnvironment

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
        self.logger.setLevel(logging.DEBUG)

        # Initialize root window
        self.root = tk.Tk()
        self.root.title("Fishing Bot Testing Interface")
        self.root.geometry("800x600")

        # Initialize components
        self.map_manager = MapManager()
        self.mock_env = MockEnvironment()
        self.mock_env.start_simulation()

        self.setup_ui()
        self.logger.info("Testing UI initialized successfully")

    def setup_ui(self):
        """Setup the UI components"""
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Left control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Position testing
        pos_frame = ttk.LabelFrame(control_frame, text="Position Testing", padding="5")
        pos_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(pos_frame, text="X:").grid(row=0, column=0, padx=5)
        self.x_var = tk.StringVar(value="100")
        ttk.Entry(pos_frame, textvariable=self.x_var, width=5).grid(row=0, column=1)

        ttk.Label(pos_frame, text="Y:").grid(row=0, column=2, padx=5)
        self.y_var = tk.StringVar(value="100")
        ttk.Entry(pos_frame, textvariable=self.y_var, width=5).grid(row=0, column=3)

        ttk.Button(pos_frame, text="Move To", 
                  command=self.test_move_to).grid(row=0, column=4, padx=5)

        # Terrain testing
        terrain_frame = ttk.LabelFrame(control_frame, text="Terrain Testing", padding="5")
        terrain_frame.pack(fill=tk.X, padx=5, pady=5)

        self.terrain_var = tk.StringVar(value="normal")
        for i, terrain in enumerate(['normal', 'water', 'mountain', 'forest']):
            ttk.Radiobutton(terrain_frame, text=terrain.capitalize(),
                          variable=self.terrain_var, value=terrain,
                          command=self.change_terrain).grid(row=i, column=0, sticky=tk.W)

        # Resource detection
        resource_frame = ttk.LabelFrame(control_frame, text="Resource Detection", padding="5")
        resource_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(resource_frame, text="Load Fish Map",
                  command=lambda: self.load_resource_map("fish")).pack(fill=tk.X, pady=2)
        ttk.Button(resource_frame, text="Load Ore Map",
                  command=lambda: self.load_resource_map("ore")).pack(fill=tk.X, pady=2)
        ttk.Button(resource_frame, text="Find Nearby",
                  command=self.find_nearby_resources).pack(fill=tk.X, pady=2)

        # Minimap preview
        preview_frame = ttk.LabelFrame(main_frame, text="Minimap Preview", padding="5")
        preview_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.minimap_label = ttk.Label(preview_frame)
        self.minimap_label.pack(expand=True, fill=tk.BOTH)

        # Status and debug info
        info_frame = ttk.LabelFrame(main_frame, text="Status & Debug Info", padding="5")
        info_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.info_text = tk.Text(info_frame, height=10, width=50)
        self.info_text.pack(expand=True, fill=tk.BOTH)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # Setup preview update
        self.update_preview()

    def update_preview(self):
        """Update minimap preview"""
        try:
            # Create test minimap
            minimap = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.rectangle(minimap, (0, 0), (200, 200), (20, 20, 20), -1)

            # Draw current position
            if self.mock_env.state.current_position:
                x, y = self.mock_env.state.current_position
                cv2.circle(minimap, (int(x), int(y)), 3, (0, 255, 0), -1)

            # Show nearby resources if available
            if hasattr(self, 'nearby_resources'):
                for resource in self.nearby_resources:
                    rx, ry = resource.position
                    cv2.circle(minimap, (int(rx), int(ry)), 5, (255, 0, 0), 1)

            # Convert to PhotoImage
            image = Image.fromarray(cv2.cvtColor(minimap, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image)

            self.minimap_label.configure(image=photo)
            self.minimap_label.image = photo  # Keep reference

        except Exception as e:
            self.logger.error(f"Error updating preview: {str(e)}")

        # Schedule next update
        self.root.after(100, self.update_preview)

    def log_info(self, message):
        """Add message to info text"""
        self.info_text.insert(tk.END, f"{message}\n")
        self.info_text.see(tk.END)

    def test_move_to(self):
        """Test movement to specified position"""
        try:
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
        """Update current terrain type"""
        terrain = self.terrain_var.get()
        success = self.mock_env.set_game_state(terrain_type=terrain)

        if success:
            self.log_info(f"Changed terrain to: {terrain}")
            self.status_var.set(f"Current terrain: {terrain}")
        else:
            self.log_info(f"Failed to change terrain to: {terrain}")

    def load_resource_map(self, resource_type):
        """Load resource map for testing"""
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
            
    def test_position_detection(self):
        """Test minimap position detection"""
        try:
            self.status_var.set("Testing position detection...")
            
            # Generate test positions
            test_positions = [
                (100, 100), (50, 50), (150, 150),
                (150, 50), (50, 150)
            ]
            
            self.info_text.delete(1.0, tk.END)
            self.log_info("Position Detection Test Results:")
            
            for pos in test_positions:
                # Update mock environment
                self.mock_env.set_game_state(current_position=pos)
                
                # Get minimap
                minimap = np.zeros((200, 200, 3), dtype=np.uint8)
                cv2.circle(minimap, pos, 3, (20, 200, 230), -1)
                
                # Detect position
                detected = self.map_manager.detect_player_position(minimap)
                if detected:
                    error = np.sqrt((detected.x - pos[0])**2 + 
                                  (detected.y - pos[1])**2)
                    result = "✓" if error < 3 else "✗"
                    self.log_info(f"{result} Position {pos}: "
                                  f"detected at ({detected.x}, {detected.y}), "
                                  f"error: {error:.1f}px")
                else:
                    self.log_info(f"✗ Position {pos}: Detection failed")
            
            self.status_var.set("Position detection test complete")
            
        except Exception as e:
            self.logger.error(f"Error in position detection test: {str(e)}")
            self.status_var.set("Error in position detection test")
            
    def test_resource_detection(self):
        """Test resource spot detection"""
        try:
            self.status_var.set("Testing resource detection...")
            
            # Load test map
            map_path = Path("maps/mase_knoll_fish.png")
            if not map_path.exists():
                self.log_info("Error: Test map not found")
                return
                
            success = self.map_manager.load_map("mase knoll fish")
            if not success:
                self.log_info("Error: Failed to load test map")
                return
            
            nodes = self.map_manager.resource_nodes.get('mase_knoll_fish', [])
            self.info_text.delete(1.0, tk.END)
            self.log_info("Resource Detection Results:")
            self.log_info(f"Detected {len(nodes)} fishing spots")
            
            for i, node in enumerate(nodes[:5]):  # Show first 5 spots
                self.log_info(f"Spot {i+1}: position {node.position}")
            
            self.status_var.set("Resource detection test complete")
            
        except Exception as e:
            self.logger.error(f"Error in resource detection test: {str(e)}")
            self.status_var.set("Error in resource detection test")
            
    def test_navigation(self):
        """Test navigation system"""
        try:
            self.status_var.set("Testing navigation...")
            
            # Test navigation between points
            start_pos = (100, 100)
            test_targets = [
                (150, 150), (50, 50), (150, 50), (50, 150)
            ]
            
            self.info_text.delete(1.0, tk.END)
            self.log_info("Navigation Test Results:")
            
            self.mock_env.set_game_state(current_position=start_pos)
            
            for target in test_targets:
                # Update mock position
                success = self.mock_env.move_to(target)
                current_pos = self.mock_env.state.current_position
                
                if success:
                    error = np.sqrt((current_pos[0] - target[0])**2 + 
                                  (current_pos[1] - target[1])**2)
                    result = "✓" if error < 5 else "✗"
                    self.log_info(f"{result} Navigation to {target}: "
                                  f"reached {current_pos}, error: {error:.1f}px")
                else:
                    self.log_info(f"✗ Navigation to {target} failed")
            
            self.status_var.set("Navigation test complete")
            
        except Exception as e:
            self.logger.error(f"Error in navigation test: {str(e)}")
            self.status_var.set("Error in navigation test")
            
    def run(self):
        """Start the UI"""
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