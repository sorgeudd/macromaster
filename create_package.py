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
        'start.bat',
    ]

    # Create dist directory
    dist_dir = Path('dist')
    dist_dir.mkdir(exist_ok=True)

    # Create ZIP file
    zip_path = dist_dir / 'FishingBot_Installer.zip'
    with zipfile.ZipFile(zip_path, 'w') as zf:
        # Add essential files
        for file in files:
            if os.path.exists(file):
                zf.write(file)
                print(f"Added {file}")
            else:
                print(f"Warning: {file} not found")

        # Create empty debug_screenshots directory in zip
        zf.writestr('debug_screenshots/.gitkeep', '')
        print("Created debug_screenshots directory")

    print(f"\nInstaller package created: {zip_path}")
    print("\nContents:")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            print(f"- {name}")

if __name__ == '__main__':
    create_package()