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

                # Enhanced visualization settings
                self.path_color = "#2196F3"  # Blue for mouse movement
                self.click_color = "#F44336"  # Red for clicks
                self.scroll_color = "#4CAF50"  # Green for scrolls
                self.key_color = "#9C27B0"  # Purple for keyboard events
                self.sound_color = "#FF9800"  # Orange for sound triggers
                self.path_width = 2
                self.point_radius = 4

                # Add timeline visualization
                self.timeline_height = 50
                self.canvas.create_rectangle(0, window_size[1]-self.timeline_height, 
                                          window_size[0], window_size[1],
                                          fill="#EEEEEE", outline="")

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
        """Create enhanced playback control panel"""
        if self.headless:
            return

        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # Speed control with label
        speed_frame = ttk.Frame(control_frame)
        speed_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
        speed_var = tk.StringVar(value="1.0")
        speed_entry = ttk.Entry(speed_frame, textvariable=speed_var, width=5)
        speed_entry.pack(side=tk.LEFT, padx=5)

        def update_speed():
            try:
                new_speed = float(speed_var.get())
                if 0.1 <= new_speed <= 10.0:
                    self.playback_speed = new_speed
                else:
                    speed_var.set("1.0")
                    self.playback_speed = 1.0
            except ValueError:
                speed_var.set("1.0")
                self.playback_speed = 1.0

        speed_entry.bind('<Return>', lambda e: update_speed())

        # Legend frame
        legend_frame = ttk.Frame(control_frame)
        legend_frame.pack(side=tk.LEFT, padx=20)

        # Add colored squares for legend
        legend_items = [
            ("Mouse", self.path_color),
            ("Click", self.click_color),
            ("Scroll", self.scroll_color),
            ("Keys", self.key_color),
            ("Sound", self.sound_color)
        ]

        for text, color in legend_items:
            item_frame = ttk.Frame(legend_frame)
            item_frame.pack(side=tk.LEFT, padx=5)
            canvas = tk.Canvas(item_frame, width=10, height=10)
            canvas.create_rectangle(0, 0, 10, 10, fill=color, outline="")
            canvas.pack(side=tk.LEFT)
            ttk.Label(item_frame, text=text).pack(side=tk.LEFT)

        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Play", 
                  command=self.start_playback).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Stop", 
                  command=self.stop_playback).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Clear", 
                  command=self.clear).pack(side=tk.LEFT, padx=2)

    def _draw_timeline_marker(self, time_position: float, total_duration: float):
        """Draw marker on timeline showing current position"""
        if self.headless:
            return

        # Clear previous timeline markers
        self.canvas.delete("timeline_marker")

        # Calculate position on timeline
        window_width = self.canvas.winfo_width()
        marker_x = (time_position / total_duration) * window_width if total_duration > 0 else 0
        timeline_y = self.canvas.winfo_height() - self.timeline_height

        # Draw marker
        self.canvas.create_line(marker_x, timeline_y, marker_x, timeline_y + self.timeline_height,
                              fill="red", width=2, tags="timeline_marker")

    def add_point(self, x: int, y: int, action_type: str = 'move', key_info: Optional[str] = None):
        """Add a point to the visualization with enhanced action types"""
        if not self.recording:
            return

        self.points.append((x, y))
        action = {
            'type': action_type,
            'x': x,
            'y': y,
            'key_info': key_info,
            'timestamp': time.time()
        }
        self.actions.append(action)

        if not self.headless:
            # Draw the new point based on action type
            if len(self.points) > 1:
                prev_x, prev_y = self.points[-2]
                if action_type == 'move':
                    self.canvas.create_line(prev_x, prev_y, x, y,
                                       fill=self.path_color,
                                       width=self.path_width,
                                       smooth=True)

            # Draw action indicators
            if action_type == 'click':
                self._draw_click(x, y)
            elif action_type == 'scroll':
                self._draw_scroll(x, y)
            elif action_type in ['keydown', 'keyup']:
                self._draw_key_event(x, y, key_info)
            elif action_type == 'sound_trigger':
                self._draw_sound_trigger(x, y)

            # Update timeline
            if self.actions:
                start_time = self.actions[0]['timestamp']
                current_time = action['timestamp']
                self._draw_timeline_marker(current_time - start_time, 
                                        max(action['timestamp'] for action in self.actions) - start_time)

            self.canvas.update()

    def _draw_key_event(self, x: int, y: int, key_info: Optional[str]):
        """Draw a keyboard event indicator"""
        if self.headless:
            return

        r = self.point_radius
        self.canvas.create_rectangle(x-r, y-r, x+r, y+r,
                                fill=self.key_color,
                                outline=self.key_color)
        if key_info:
            self.canvas.create_text(x, y-r-5, text=key_info,
                               fill=self.key_color,
                               font=('TkDefaultFont', 8))

    def _draw_sound_trigger(self, x: int, y: int):
        """Draw a sound trigger indicator"""
        if self.headless:
            return

        r = self.point_radius * 1.5
        points = [
            (x, y-r),  # top
            (x+r, y),  # right
            (x, y+r),  # bottom
            (x-r, y)   # left
        ]
        self.canvas.create_polygon(points,
                              fill=self.sound_color,
                              outline=self.sound_color)

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

                self.add_point(action['x'], action['y'], action['type'], action.get('key_info'))

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
                self.add_point(action['x'], action['y'], action['type'], action.get('key_info'))

            self.logger.info(f"Loaded visualization from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading visualization: {e}")
            return False