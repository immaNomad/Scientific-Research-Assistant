#!/bin/bash
# Installation script for RAG-Enhanced Scientific Research Assistant

echo "🚀 Setting up RAG-Enhanced Scientific Research Assistant..."
echo "============================================================"

# Check if Python 3.8+ is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python $PYTHON_VERSION detected"

# Check for minimum Python version (3.8)
if python3 -c 'import sys; exit(not (sys.version_info >= (3, 8)))'; then
    echo "✅ Python version is compatible"
else
    echo "❌ Python 3.8 or higher is required"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
else
    echo "❌ Error installing dependencies"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/{embeddings,cache,papers,models,feedback} logs
echo "✅ Directories created!"

# Create .env file
echo "📝 Setting up configuration..."
if [ ! -f ".env" ]; then
    cp env.example .env
    echo "✅ Created .env file from template"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "============================================================"
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit the .env file with your API keys (optional but recommended)"
echo "2. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo "3. Run the application:"
echo "   python main.py"
echo "   or"
echo "   python main.py search 'machine learning transformers'"
echo ""
echo "💡 For help, run: python main.py --help"
echo ""
echo "⚠️  Important: Remember to activate the virtual environment before using the tool!"
echo "   Run: source venv/bin/activate" 