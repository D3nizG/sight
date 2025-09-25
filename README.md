# Sight 👁️

Real-time camera app with computer vision features using OpenCV and MediaPipe.

## Features

- 📹 **Real-time camera feed** with customizable resolution
- 🖐️ **Hand landmark detection** using MediaPipe (21 points per hand)
- 🎨 **Six image processing modes**: grayscale, blur, edge detection, binary threshold, and Sobel gradients
- ⌨️ **Interactive controls** for real-time mode switching
- 💾 **Frame capture** with timestamp naming
- 📊 **FPS monitoring** with overlay display
- 🪞 **Mirror mode** (horizontally flipped) by default

## Quick Start

**Option 1: Automatic Setup**
```bash
./setup.sh
```

**Option 2: Manual Setup**
```bash
# Requires Python 3.11 (for MediaPipe compatibility)
brew install python@3.11
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
source venv/bin/activate
python -m sight.camera_app --camera-index 0 --width 1280 --height 720
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `m` | Cycle through processing modes |
| `h` | Toggle hand landmark detection |
| `s` | Save current frame to `output/` directory |

## Processing Modes

1. **None** - Original camera feed
2. **Grayscale** - Black and white conversion
3. **Blur** - Gaussian blur effect
4. **Edges** - Canny edge detection with overlay
5. **Binary** - Adaptive threshold (black/white)
6. **Sobel** - Sobel gradient edge detection

## CLI Flags

- --camera-index, -c: camera device index (default 0)
- --width, -W: frame width (default 1280)
- --height, -H: frame height (default 720)
- --mode, -m: initial processing mode
- --no-flip: disable horizontal flip (mirror)
- --no-fps: hide FPS overlay
- --hands / --no-hands: enable/disable Mediapipe hands
- --hand-max: max number of hands (default 2)
- --hand-det: min detection confidence (default 0.6)
- --hand-track: min tracking confidence (default 0.6)

## System Requirements

- **Python 3.11** (required for MediaPipe compatibility)
- **macOS/Linux/Windows** (tested on macOS)
- **Webcam** or compatible camera device
- **OpenGL support** for MediaPipe hand tracking

## Dependencies

- `opencv-python` - Computer vision and camera capture
- `mediapipe` - Hand landmark detection
- `numpy` - Array operations
- Additional dependencies listed in `requirements.txt`

## Troubleshooting

**Camera Permission Issues (macOS)**
- Go to System Preferences → Privacy & Security → Camera
- Enable camera access for your terminal application

**MediaPipe Installation Issues**
- Ensure you're using Python 3.11 (not 3.13)
- Try reinstalling with: `pip install --upgrade mediapipe`

**No Camera Found**
- Try different camera indices: `--camera-index 1`, `--camera-index 2`
- Check if other applications are using the camera
