"""Windows-based macro and sound recording interface"""
import tkinter as tk
from tkinter import ttk
import keyboard
import logging
from sound_macro_manager import SoundMacroManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('macro_interface.log')
    ]
)
logger = logging.getLogger('MacroInterface')

class MacroInterface:
    def __init__(self):
        self.logger = logger
        self.sound_manager = SoundMacroManager()

        # Initialize main window
        self.root = tk.Tk()
        self.root.title("Macro Management")
        self.root.geometry("600x800")

        # Setup interface
        self.setup_gui()

        # Recording states
        self.is_recording_macro = False
        self.is_recording_sound = False
        self.is_monitoring = False
        self.is_recording_hotkey = False
        self.current_hotkey = None

        # Start hotkey listener
        self.setup_hotkey_listener()

    def setup_hotkey_listener(self):
        """Setup global hotkey listener"""
        def on_hotkey(event):
            if not self.is_recording_hotkey:
                # Only trigger macros if we're not recording a new hotkey
                hotkey = event.name
                self.sound_manager.trigger_macro_by_hotkey(hotkey)

        keyboard.on_press(on_hotkey)

    def setup_gui(self):
        """Setup the GUI elements"""
        # Status Frame
        status_frame = ttk.LabelFrame(self.root, text="Status", padding="5")
        status_frame.pack(fill=tk.X, padx=5, pady=5)

        self.macro_status = ttk.Label(status_frame, text="Macro Status: Ready")
        self.macro_status.pack(side=tk.LEFT, padx=5)

        self.sound_status = ttk.Label(status_frame, text="Sound Status: Ready")
        self.sound_status.pack(side=tk.LEFT, padx=5)

        # Macro Control Frame
        macro_frame = ttk.LabelFrame(self.root, text="Macro Management", padding="5")
        macro_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(macro_frame, text="Macro Name:").pack(fill=tk.X)
        self.macro_name = ttk.Entry(macro_frame)
        self.macro_name.pack(fill=tk.X, padx=5, pady=2)

        # Hotkey assignment frame
        hotkey_frame = ttk.Frame(macro_frame)
        hotkey_frame.pack(fill=tk.X, padx=5, pady=2)

        self.hotkey_label = ttk.Label(hotkey_frame, text="Hotkey: None")
        self.hotkey_label.pack(side=tk.LEFT, padx=5)

        self.assign_hotkey_btn = ttk.Button(hotkey_frame, text="Assign Hotkey", command=self.start_hotkey_assignment)
        self.assign_hotkey_btn.pack(side=tk.LEFT, padx=5)

        self.clear_hotkey_btn = ttk.Button(hotkey_frame, text="Clear Hotkey", command=self.clear_hotkey)
        self.clear_hotkey_btn.pack(side=tk.LEFT, padx=5)

        self.record_macro_btn = ttk.Button(macro_frame, text="Record Macro", command=self.start_macro_recording)
        self.record_macro_btn.pack(fill=tk.X, padx=5, pady=2)

        self.stop_macro_btn = ttk.Button(macro_frame, text="Stop Recording", command=self.stop_macro_recording, state='disabled')
        self.stop_macro_btn.pack(fill=tk.X, padx=5, pady=2)

        self.play_macro_btn = ttk.Button(macro_frame, text="Play Macro", command=self.play_macro)
        self.play_macro_btn.pack(fill=tk.X, padx=5, pady=2)

        # Sound Control Frame
        sound_frame = ttk.LabelFrame(self.root, text="Sound Management", padding="5")
        sound_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(sound_frame, text="Sound Name:").pack(fill=tk.X)
        self.sound_name = ttk.Entry(sound_frame)
        self.sound_name.pack(fill=tk.X, padx=5, pady=2)

        self.record_sound_btn = ttk.Button(sound_frame, text="Record Sound", command=self.start_sound_recording)
        self.record_sound_btn.pack(fill=tk.X, padx=5, pady=2)

        self.stop_sound_btn = ttk.Button(sound_frame, text="Stop Recording", command=self.stop_sound_recording, state='disabled')
        self.stop_sound_btn.pack(fill=tk.X, padx=5, pady=2)

        self.play_sound_btn = ttk.Button(sound_frame, text="Play Sound", command=self.play_sound)
        self.play_sound_btn.pack(fill=tk.X, padx=5, pady=2)

        self.monitor_btn = ttk.Button(sound_frame, text="Start Monitoring", command=self.toggle_monitoring)
        self.monitor_btn.pack(fill=tk.X, padx=5, pady=2)

        # Log Frame
        log_frame = ttk.LabelFrame(self.root, text="System Logs", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_text = tk.Text(log_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def add_log(self, message, level="INFO"):
        """Add a message to the log window"""
        self.log_text.insert(tk.END, f"{level}: {message}\n")
        self.log_text.see(tk.END)

    def start_macro_recording(self):
        """Start recording a macro"""
        macro_name = self.macro_name.get().strip()
        if not macro_name:
            self.add_log("Please enter a macro name", "ERROR")
            return

        try:
            if self.sound_manager.record_macro(macro_name):
                self.is_recording_macro = True
                self.record_macro_btn.config(state='disabled')
                self.stop_macro_btn.config(state='normal')
                self.macro_status.config(text="Macro Status: Recording")
                self.add_log(f"Started recording macro: {macro_name}")
        except Exception as e:
            self.add_log(f"Error starting macro recording: {e}", "ERROR")

    def stop_macro_recording(self):
        """Stop recording a macro"""
        try:
            if self.sound_manager.stop_macro_recording():
                self.is_recording_macro = False
                self.record_macro_btn.config(state='normal')
                self.stop_macro_btn.config(state='disabled')
                self.macro_status.config(text="Macro Status: Ready")
                self.add_log("Macro recording completed")
        except Exception as e:
            self.add_log(f"Error stopping macro recording: {e}", "ERROR")

    def play_macro(self):
        """Play a recorded macro"""
        macro_name = self.macro_name.get().strip()
        if not macro_name:
            self.add_log("Please enter a macro name", "ERROR")
            return

        self.add_log(f"Playing macro: {macro_name}")
        # Implementation will be added once we have the macro playback functionality

    def start_sound_recording(self):
        """Start recording a sound trigger"""
        sound_name = self.sound_name.get().strip()
        if not sound_name:
            self.add_log("Please enter a sound name", "ERROR")
            return

        try:
            if self.sound_manager.record_sound_trigger(sound_name):
                self.is_recording_sound = True
                self.record_sound_btn.config(state='disabled')
                self.stop_sound_btn.config(state='normal')
                self.sound_status.config(text="Sound Status: Recording")
                self.add_log(f"Started recording sound: {sound_name}")
        except Exception as e:
            self.add_log(f"Error starting sound recording: {e}", "ERROR")

    def stop_sound_recording(self):
        """Stop recording a sound trigger"""
        try:
            if self.sound_manager.stop_recording():
                self.is_recording_sound = False
                self.record_sound_btn.config(state='normal')
                self.stop_sound_btn.config(state='disabled')
                self.sound_status.config(text="Sound Status: Ready")
                self.add_log("Sound recording completed")
        except Exception as e:
            self.add_log(f"Error stopping sound recording: {e}", "ERROR")

    def play_sound(self):
        """Play a recorded sound"""
        sound_name = self.sound_name.get().strip()
        if not sound_name:
            self.add_log("Please enter a sound name", "ERROR")
            return

        self.add_log(f"Playing sound: {sound_name}")
        # Implementation will be added once we have the sound playback functionality

    def toggle_monitoring(self):
        """Toggle sound monitoring"""
        try:
            if not self.is_monitoring:
                if self.sound_manager.start_monitoring():
                    self.is_monitoring = True
                    self.monitor_btn.config(text="Stop Monitoring")
                    self.sound_status.config(text="Sound Status: Monitoring")
                    self.add_log("Started sound monitoring")
            else:
                self.sound_manager.stop_monitoring()
                self.is_monitoring = False
                self.monitor_btn.config(text="Start Monitoring")
                self.sound_status.config(text="Sound Status: Ready")
                self.add_log("Stopped sound monitoring")
        except Exception as e:
            self.add_log(f"Error toggling monitoring: {e}", "ERROR")

    def start_hotkey_assignment(self):
        """Start recording a new hotkey assignment"""
        macro_name = self.macro_name.get().strip()
        if not macro_name:
            self.add_log("Please enter a macro name", "ERROR")
            return

        self.is_recording_hotkey = True
        self.assign_hotkey_btn.config(text="Press any key...", state='disabled')
        self.add_log("Press any key to assign as hotkey...")

        def on_key(event):
            if self.is_recording_hotkey:
                self.is_recording_hotkey = False
                hotkey = event.name
                if self.sound_manager.assign_hotkey(macro_name, hotkey):
                    self.current_hotkey = hotkey
                    self.hotkey_label.config(text=f"Hotkey: {hotkey}")
                    self.add_log(f"Assigned hotkey '{hotkey}' to macro '{macro_name}'")
                else:
                    self.add_log("Failed to assign hotkey", "ERROR")

                self.assign_hotkey_btn.config(text="Assign Hotkey", state='normal')
                keyboard.unhook(on_key)

        keyboard.on_press(on_key)

    def clear_hotkey(self):
        """Clear the currently assigned hotkey"""
        macro_name = self.macro_name.get().strip()
        if not macro_name:
            self.add_log("Please enter a macro name", "ERROR")
            return

        if self.sound_manager.remove_hotkey(macro_name):
            self.current_hotkey = None
            self.hotkey_label.config(text="Hotkey: None")
            self.add_log(f"Cleared hotkey for macro '{macro_name}'")
        else:
            self.add_log("No hotkey assigned to clear")

    def update_hotkey_display(self):
        """Update the hotkey display for the current macro"""
        macro_name = self.macro_name.get().strip()
        if macro_name:
            hotkey = self.sound_manager.get_macro_hotkey(macro_name)
            if hotkey:
                self.current_hotkey = hotkey
                self.hotkey_label.config(text=f"Hotkey: {hotkey}")
            else:
                self.current_hotkey = None
                self.hotkey_label.config(text="Hotkey: None")


    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":
    try:
        app = MacroInterface()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        logger.error(logging.traceback.format_exc())