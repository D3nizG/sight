#!/bin/bash
# Setup script for Sight camera app

set -e

echo "🎯 Setting up Sight camera app..."

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Python 3.11 is required but not found."
    echo "Install it with: brew install python@3.11"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment with Python 3.11..."
python3.11 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "To run the app:"
echo "  source venv/bin/activate"
echo "  python -m sight.camera_app --camera-index 0 --width 1280 --height 720"
echo ""
echo "Keyboard controls:"
echo "  q: quit"
echo "  m: cycle processing modes"
echo "  h: toggle hand landmarks"
echo "  s: save current frame"