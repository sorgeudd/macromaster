"""Main entry point for the fishing bot application"""
import tkinter as tk
from tkinter import ttk
from gui_components import MainWindow
from logger import setup_logger
import platform
import sys
import logging
import traceback

def main():
    try:
        # Setup logging
        logger = setup_logger()
        logger.info("Starting Fishing Bot")

        # Check if running in test mode
        test_mode = platform.system() != 'Windows'
        if test_mode:
            logger.warning("Running in test mode (non-Windows platform)")
            logger.debug("Debug logging enabled for test mode")

        # Create main window
        root = tk.Tk()
        root.title("Fishing Bot")

        # Set a larger window size to better accommodate all components
        root.geometry("500x1000")  # Increased height to 1000px
        logger.debug("Created main window with geometry 500x1000")

        # Create a custom style for the emergency stop button
        style = ttk.Style()
        style.configure("Emergency.TButton", foreground="red", font=('bold'))
        logger.debug("Configured custom button styles")

        # Initialize main application window
        try:
            logger.debug("Initializing MainWindow with test_mode=" + str(test_mode))
            app = MainWindow(root, test_mode=test_mode)
            logger.info("MainWindow initialized successfully")
            root.mainloop()
        except Exception as e:
            logger.error(f"Application crashed: {str(e)}")
            logger.error("Traceback: " + traceback.format_exc())
            print(f"Error: {str(e)}")
            print("Please check the log file for details")
            sys.exit(1)

    except Exception as e:
        print(f"Critical error during startup: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()