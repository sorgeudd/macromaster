# Game Automation Bot

A Python-based game automation bot with advanced computer vision capabilities for precise resource detection and automation in Windows game environments.

## Key Components
- Computer vision-based resource recognition
- Direct input handling for Windows
- Modular design for game-specific automation
- Advanced image processing and detection algorithms

## Testing Interface

### Quick Start
1. Extract all components from testing_package.zip
2. Install required packages:
```bash
pip install opencv-python pillow numpy
```
3. Run the testing interface:
```bash
python testing_ui.py
```

### Testing Features
1. Position Testing
   - Enter X and Y coordinates
   - Click "Move To" to test position detection
   - Real-time preview shows detected position

2. Terrain Testing
   - Select different terrain types (normal, water, mountain, forest)
   - Tests movement speed and behavior in different terrains

3. Resource Detection
   - Load fish or ore map
   - "Find Nearby" detects resources within range
   - Blue dots on preview show detected resources

### Debug Information
- Status bar shows current operation status
- Debug info panel shows detailed test results
- Debug images saved to disk for analysis:
  - arrow_detection_debug.png: Shows detected player position
  - resource_detection_debug.png: Shows detected resources

### Logging
All components write detailed logs to their respective files:
- testing_ui.log: Testing interface logs
- map_manager.log: Map and resource detection logs
- mock_environment.log: Environment simulation logs

## Development
- Python 3.8 or later required
- Uses OpenCV for image processing
- Tkinter for testing interface
- Numpy for numerical operations

## Troubleshooting
1. Position Detection Issues
   - Check arrow_detection_debug.png for visual feedback
   - Verify coordinates in debug info panel
   - Ensure minimap preview is updating

2. Resource Detection Issues
   - Verify map files are in maps/ directory
   - Check resource_detection_debug.png
   - Review map_manager.log for detection details

3. UI Issues
   - Check testing_ui.log for errors
   - Verify all required packages are installed
   - Restart interface if preview freezes

## Testing Workflow
1. Start with basic position detection
2. Test different terrain types
3. Load and test resource maps
4. Verify nearby resource detection
5. Monitor real-time updates in preview

For issues and feature requests, please file an issue in the GitHub repository.