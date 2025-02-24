"""Script to create distribution package"""
import os
import zipfile
from pathlib import Path

def create_package():
    # Files to include
    files = [
        'gui_interface.py',
        'sound_trigger.py',
        'test_macro.py',
        'direct_input.py',
        'install.bat',
        'start_app.bat',
        'README.md'
    ]
    
    # Create dist directory
    dist_dir = Path('dist')
    dist_dir.mkdir(exist_ok=True)
    
    # Create ZIP file
    zip_path = dist_dir / 'sound_macro_app.zip'
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for file in files:
            if os.path.exists(file):
                zf.write(file)
            else:
                print(f"Warning: {file} not found")
                
    print(f"\nPackage created: {zip_path}")
    print("Contents:")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            print(f"- {name}")

if __name__ == '__main__':
    create_package()
