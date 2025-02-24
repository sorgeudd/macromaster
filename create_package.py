"""Script to create distribution package"""
import os
import zipfile
from pathlib import Path

def create_package():
    # Files to include
    files = [
        'gui_interface.py',
        'sound_trigger.py',
        'direct_input.py',
        'macro_visualizer.py',
        'sound_macro_manager.py',
        'simple_install.bat',
        'simple_start.bat',
        'README.md',
        'gui_components.py',
        'logger.py',
        'test_macro.py',
        'config_manager.py',
    ]

    # Create dist directory
    dist_dir = Path('dist')
    dist_dir.mkdir(exist_ok=True)

    # Create ZIP file
    zip_path = dist_dir / 'sound_macro_app.zip'
    with zipfile.ZipFile(zip_path, 'w') as zf:
        # Add essential Python files and batch scripts
        for file in files:
            if os.path.exists(file):
                zf.write(file)
                print(f"Added {file}")
            else:
                print(f"Warning: {file} not found")

        # Add directories with their contents
        directories = ['macros', 'models', 'sounds']
        for directory in directories:
            if os.path.exists(directory):
                for root, _, filenames in os.walk(directory):
                    for filename in filenames:
                        file_path = os.path.join(root, filename)
                        zf.write(file_path)
                        print(f"Added {file_path}")
            else:
                print(f"Warning: {directory} directory not found")

    print(f"\nPackage created: {zip_path}")
    print("Contents:")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            print(f"- {name}")

if __name__ == '__main__':
    create_package()