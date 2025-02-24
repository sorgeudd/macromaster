"""Verify project structure and files before GitHub push"""
import os
from pathlib import Path

REQUIRED_FILES = [
    'gui_interface.py',
    'sound_macro_manager.py',
    'sound_trigger.py',
    'direct_input.py',
    'test_macro.py',
    'install.bat',
    'start_app.bat',
    'README.md'
]

REQUIRED_DIRS = [
    'macros'  # For storing recorded macros
]

def check_structure():
    """Check if all required files and directories exist"""
    missing_files = []
    missing_dirs = []
    
    # Check files
    for file in REQUIRED_FILES:
        if not Path(file).exists():
            missing_files.append(file)
    
    # Check directories
    for dir_name in REQUIRED_DIRS:
        if not Path(dir_name).exists():
            missing_dirs.append(dir_name)
            
    return missing_files, missing_dirs

def main():
    print("Checking project structure...")
    missing_files, missing_dirs = check_structure()
    
    if missing_files:
        print("\nMissing files:")
        for file in missing_files:
            print(f"- {file}")
    
    if missing_dirs:
        print("\nMissing directories:")
        for dir_name in missing_dirs:
            print(f"- {dir_name}")
            # Create missing directories
            Path(dir_name).mkdir(exist_ok=True)
            print(f"  Created directory: {dir_name}")
    
    if not missing_files and not missing_dirs:
        print("\nAll required files and directories are present!")
        return True
    
    return False

if __name__ == "__main__":
    main()
