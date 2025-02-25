"""Script to create distribution package"""
import os
import zipfile
import shutil
from pathlib import Path

def create_package():
    # Files to include
    files = [
        'main.py',
        'config.py',
        'README.md',
        'install.bat',
        'start_bot.bat',
    ]

    # Create dist directory
    dist_dir = Path('dist')
    dist_dir.mkdir(exist_ok=True)

    # Create ZIP file
    zip_path = dist_dir / 'fishing_bot.zip'
    with zipfile.ZipFile(zip_path, 'w') as zf:
        # Add essential Python files and batch scripts
        for file in files:
            if os.path.exists(file):
                zf.write(file)
                print(f"Added {file}")
            else:
                print(f"Warning: {file} not found")

        # Add required directories with their contents
        directories = ['debug_screenshots']
        for directory in directories:
            if os.path.exists(directory):
                for root, _, filenames in os.walk(directory):
                    for filename in filenames:
                        file_path = os.path.join(root, filename)
                        zf.write(file_path)
                        print(f"Added {file_path}")
            else:
                # Create empty directory in zip
                zf.writestr(f"{directory}/.gitkeep", "")
                print(f"Created empty directory: {directory}")

    print(f"\nPackage created: {zip_path}")
    print("Contents:")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            print(f"- {name}")

if __name__ == '__main__':
    create_package()