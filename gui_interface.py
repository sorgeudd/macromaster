"""GUI interface for sound macro application"""
import tkinter as tk
from tkinter import ttk, messagebox
import logging
from sound_macro_manager import SoundMacroManager
import platform
import sys

class SoundMacroGUI:
    def __init__(self):
        self.logger = logging.getLogger('SoundMacroGUI')

        # Initialize manager
        try:
            self.test_mode = platform.system() != 'Windows'
            self.manager = SoundMacroManager(test_mode=self.test_mode)

            # Create main window
            self.root = tk.Tk()
            self.root.title("Sound Macro Recorder")
            self.root.geometry("600x400")

            # Add status bar
            self.status_var = tk.StringVar(value="Ready")
            self.status_bar = ttk.Label(self.root, textvariable=self.status_var)
            self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

            # Create main frame
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Sound recording section
            sound_frame = ttk.LabelFrame(main_frame, text="Sound Triggers", padding="5")
            sound_frame.pack(fill=tk.X, padx=5, pady=5)

            ttk.Label(sound_frame, text="Sound Name:").pack(side=tk.LEFT, padx=5)
            self.sound_name = ttk.Entry(sound_frame, width=20)
            self.sound_name.pack(side=tk.LEFT, padx=5)

            ttk.Button(sound_frame, text="Record Sound (2s)", 
                      command=self.record_sound).pack(side=tk.LEFT, padx=5)

            # Macro recording section
            macro_frame = ttk.LabelFrame(main_frame, text="Macros", padding="5")
            macro_frame.pack(fill=tk.X, padx=5, pady=5)

            ttk.Label(macro_frame, text="Macro Name:").pack(side=tk.LEFT, padx=5)
            self.macro_name = ttk.Entry(macro_frame, width=20)
            self.macro_name.pack(side=tk.LEFT, padx=5)

            ttk.Button(macro_frame, text="Record Macro (5s)", 
                      command=self.record_macro).pack(side=tk.LEFT, padx=5)

            # Mapping section
            map_frame = ttk.LabelFrame(main_frame, text="Map Sound to Macro", padding="5")
            map_frame.pack(fill=tk.X, padx=5, pady=5)

            # Sound selection
            ttk.Label(map_frame, text="Sound:").pack(side=tk.LEFT, padx=5)
            self.sound_var = tk.StringVar()
            self.sound_combo = ttk.Combobox(map_frame, textvariable=self.sound_var, width=15)
            self.sound_combo.pack(side=tk.LEFT, padx=5)

            ttk.Label(map_frame, text="Macro:").pack(side=tk.LEFT, padx=5)
            self.macro_var = tk.StringVar()
            self.macro_combo = ttk.Combobox(map_frame, textvariable=self.macro_var, width=15)
            self.macro_combo.pack(side=tk.LEFT, padx=5)

            ttk.Button(map_frame, text="Map", 
                      command=self.map_sound_to_macro).pack(side=tk.LEFT, padx=5)

            # Monitoring controls
            control_frame = ttk.Frame(main_frame)
            control_frame.pack(fill=tk.X, padx=5, pady=5)

            self.monitor_btn = ttk.Button(control_frame, text="Start Monitoring", 
                                        command=self.toggle_monitoring)
            self.monitor_btn.pack(side=tk.LEFT, padx=5)

            self.monitoring = False
            self.update_lists()

            # Add test mode indicator if active
            if self.test_mode:
                test_label = ttk.Label(self.root, text="Running in Test Mode", 
                                     foreground="red")
                test_label.pack(side=tk.BOTTOM, pady=5)

        except Exception as e:
            self.logger.error(f"Failed to initialize GUI: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize: {str(e)}")
            sys.exit(1)

    def update_status(self, message):
        """Update status bar message"""
        self.status_var.set(message)
        self.root.update()

    def update_lists(self):
        """Update combobox lists"""
        try:
            sounds = self.manager.sound_trigger.get_trigger_names()
            self.sound_combo['values'] = sounds

            macros = [p.stem for p in self.manager.macros_dir.glob('*.json')]
            self.macro_combo['values'] = macros
        except Exception as e:
            self.logger.error(f"Error updating lists: {str(e)}")
            self.update_status("Error updating lists")

    def record_sound(self):
        """Record a new sound trigger"""
        name = self.sound_name.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a sound name")
            return

        try:
            self.update_status(f"Recording sound '{name}' in 3 seconds...")
            self.root.update()
            import time
            time.sleep(3)

            if self.manager.record_sound_trigger(name):
                self.update_status(f"Successfully recorded sound '{name}'")
                self.sound_name.delete(0, tk.END)
                self.update_lists()
            else:
                self.update_status("Failed to record sound")
        except Exception as e:
            self.logger.error(f"Error recording sound: {str(e)}")
            self.update_status("Error recording sound")
            messagebox.showerror("Error", f"Failed to record sound: {str(e)}")

    def record_macro(self):
        """Record a new macro"""
        name = self.macro_name.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a macro name")
            return

        try:
            self.update_status(f"Recording macro '{name}' in 3 seconds...")
            self.root.update()
            import time
            time.sleep(3)

            if self.manager.record_macro(name, duration=5.0):
                self.update_status(f"Successfully recorded macro '{name}'")
                self.macro_name.delete(0, tk.END)
                self.update_lists()
            else:
                self.update_status("Failed to record macro")
        except Exception as e:
            self.logger.error(f"Error recording macro: {str(e)}")
            self.update_status("Error recording macro")
            messagebox.showerror("Error", f"Failed to record macro: {str(e)}")

    def map_sound_to_macro(self):
        """Map selected sound to selected macro"""
        sound = self.sound_var.get()
        macro = self.macro_var.get()

        if not sound or not macro:
            messagebox.showerror("Error", "Please select both sound and macro")
            return

        try:
            if self.manager.assign_macro_to_sound(sound, macro):
                self.update_status(f"Mapped sound '{sound}' to macro '{macro}'")
            else:
                self.update_status("Failed to map sound to macro")
        except Exception as e:
            self.logger.error(f"Error mapping sound to macro: {str(e)}")
            self.update_status("Error mapping sound to macro")
            messagebox.showerror("Error", f"Failed to map sound to macro: {str(e)}")

    def toggle_monitoring(self):
        """Toggle sound monitoring"""
        try:
            if not self.monitoring:
                if self.manager.start_monitoring():
                    self.monitoring = True
                    self.monitor_btn.configure(text="Stop Monitoring")
                    self.update_status("Monitoring started")
                else:
                    self.update_status("Failed to start monitoring")
            else:
                self.manager.stop_monitoring()
                self.monitoring = False
                self.monitor_btn.configure(text="Start Monitoring")
                self.update_status("Monitoring stopped")
        except Exception as e:
            self.logger.error(f"Error toggling monitoring: {str(e)}")
            self.update_status("Error toggling monitoring")
            messagebox.showerror("Error", f"Error toggling monitoring: {str(e)}")

    def run(self):
        """Start the GUI"""
        try:
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"Error in GUI mainloop: {str(e)}")
            messagebox.showerror("Error", f"Application error: {str(e)}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Start GUI
    gui = SoundMacroGUI()
    gui.run()