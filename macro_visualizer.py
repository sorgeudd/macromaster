"""Visualization module for macro recording preview"""
import tkinter as tk
from tkinter import ttk
import time
import math
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path
import logging
import platform

class MacroVisualizer:
    def __init__(self, window_size: Tuple[int, int] = (800, 600), headless: bool = False):
        self.logger = logging.getLogger('MacroVisualizer')
        self.headless = headless or platform.system() != 'Windows'

        try:
            if not self.headless:
                # Create visualization window
                self.root = tk.Toplevel()
                self.root.title("Macro Preview")
                self.root.geometry(f"{window_size[0]}x{window_size[1]}")

                # Create canvas for drawing
                self.canvas = tk.Canvas(self.root, width=window_size[0], height=window_size[1],
                                   background='white')
                self.canvas.pack(fill=tk.BOTH, expand=True)

                # Path visualization settings
                self.path_color = "#2196F3"  # Blue
                self.click_color = "#F44336"  # Red
                self.scroll_color = "#4CAF50"  # Green
                self.path_width = 2
                self.point_radius = 4

                # Playback controls
                self.playback_speed = 1.0
                self._create_controls()

                self.logger.info("GUI visualization initialized")
            else:
                self.logger.info("Running in headless mode")
        except Exception as e:
            self.logger.error(f"Error initializing visualization: {e}")
            self.headless = True

        # Store recorded points and actions (works in both modes)
        self.points: List[Tuple[int, int]] = []
        self.actions: List[Dict] = []
        self.recording = False
        self.playback_running = False

    def _create_controls(self):
        """Create playback control panel"""
        if self.headless:
            return

        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # Speed control
        ttk.Label(control_frame, text="Speed:").pack(side=tk.LEFT)
        speed_var = tk.StringVar(value="1.0")
        speed_entry = ttk.Entry(control_frame, textvariable=speed_var, width=5)
        speed_entry.pack(side=tk.LEFT, padx=5)

        def update_speed():
            try:
                self.playback_speed = float(speed_var.get())
            except ValueError:
                speed_var.set("1.0")
                self.playback_speed = 1.0

        speed_entry.bind('<Return>', lambda e: update_speed())

        # Control buttons
        ttk.Button(control_frame, text="Play", 
                  command=self.start_playback).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop", 
                  command=self.stop_playback).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Clear", 
                  command=self.clear).pack(side=tk.LEFT, padx=5)

    def start_recording(self):
        """Start recording visualization"""
        self.recording = True
        self.clear()
        self.logger.info("Started recording visualization")

    def stop_recording(self):
        """Stop recording visualization"""
        self.recording = False
        self.logger.info("Stopped recording visualization")

    def add_point(self, x: int, y: int, action_type: str = 'move'):
        """Add a point to the visualization"""
        if not self.recording:
            return

        self.points.append((x, y))
        self.actions.append({
            'type': action_type,
            'x': x,
            'y': y,
            'timestamp': time.time()
        })

        if not self.headless:
            # Draw the new point
            if len(self.points) > 1:
                prev_x, prev_y = self.points[-2]
                self.canvas.create_line(prev_x, prev_y, x, y,
                                   fill=self.path_color,
                                   width=self.path_width,
                                   smooth=True)

            # Draw action indicators
            if action_type == 'click':
                self._draw_click(x, y)
            elif action_type == 'scroll':
                self._draw_scroll(x, y)

            self.canvas.update()

    def _draw_click(self, x: int, y: int):
        """Draw a click indicator"""
        if self.headless:
            return

        r = self.point_radius
        self.canvas.create_oval(x-r, y-r, x+r, y+r,
                           fill=self.click_color,
                           outline=self.click_color)

    def _draw_scroll(self, x: int, y: int):
        """Draw a scroll indicator"""
        if self.headless:
            return

        r = self.point_radius
        self.canvas.create_rectangle(x-r, y-r, x+r, y+r,
                                fill=self.scroll_color,
                                outline=self.scroll_color)

    def clear(self):
        """Clear the visualization"""
        if not self.headless:
            self.canvas.delete("all")
        self.points = []
        self.actions = []
        self.logger.debug("Cleared visualization")

    def start_playback(self):
        """Start macro playback visualization"""
        if self.headless or not self.actions or self.playback_running:
            return

        self.playback_running = True
        self.clear()

        def animate():
            if not self.playback_running:
                return

            for i, action in enumerate(self.actions):
                if not self.playback_running:
                    break

                self.add_point(action['x'], action['y'], action['type'])

                if i < len(self.actions) - 1:
                    delay = (self.actions[i+1]['timestamp'] - action['timestamp']) / self.playback_speed
                    self.root.after(int(delay * 1000), animate)
                    break
            else:
                self.playback_running = False

        animate()
        self.logger.info("Started playback visualization")

    def stop_playback(self):
        """Stop macro playback visualization"""
        self.playback_running = False
        self.logger.info("Stopped playback visualization")

    def save_visualization(self, filepath: str):
        """Save the current visualization data"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.actions, f, indent=2)
            self.logger.info(f"Saved visualization to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving visualization: {e}")
            return False

    def load_visualization(self, filepath: str):
        """Load visualization data from file"""
        try:
            with open(filepath, 'r') as f:
                actions = json.load(f)

            self.clear()
            for action in actions:
                self.add_point(action['x'], action['y'], action['type'], action['timestamp'])

            self.logger.info(f"Loaded visualization from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading visualization: {e}")
            return False