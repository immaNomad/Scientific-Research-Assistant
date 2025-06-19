# Installation Guide

This guide will help you set up the Scientific Research Assistant on your system.

## 📋 System Requirements

### Operating System
- **Linux**: Ubuntu 18.04+ / CentOS 7+ / Similar distributions
- **Windows**: Windows 10/11 (with WSL recommended)
- **macOS**: macOS 10.15+

### Hardware Requirements
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: 2GB free space
- **Internet**: Stable connection for API access

### Software Prerequisites
- **Python**: 3.8 or higher
- **pip**: Python package manager
- **Git**: For cloning repository

## 🚀 Installation Methods

### Method 1: Quick Setup (Recommended)

1. **Download and Extract**
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd MiniProjectEmerg
   ```

2. **Run Installation Script**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

3. **Configure API Key**
   ```bash
   cp env.example .env
   nano .env  # Add your OPENAI_API_KEY
   ```

4. **Launch Application**
   ```bash
   python3 launch_gui.py
   ```

### Method 2: Manual Installation

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd MiniProjectEmerg
   ```

2. **Create Virtual Environment (Optional but Recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables**
   ```bash
   cp env.example .env
   ```
   
   Edit `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   SEMANTIC_SCHOLAR_API_KEY=optional_semantic_scholar_key
   PUBMED_API_KEY=optional_pubmed_key
   ```

5. **Test Installation**
   ```bash
   python3 launch_gui.py
   ```

## 🔑 API Key Setup

### OpenAI API Key (Required)

1. **Create OpenAI Account**
   - Visit [OpenAI Platform](https://platform.openai.com/)
   - Sign up or log in to your account

2. **Generate API Key**
   - Go to [API Keys](https://platform.openai.com/api-keys)
   - Click "Create new secret key"
   - Copy the generated key

3. **Add to Environment**
   ```bash
   # In your .env file
   OPENAI_API_KEY=sk-your-actual-key-here
   ```

### Optional API Keys

#### Semantic Scholar (Higher Rate Limits)
1. **Register**: [Semantic Scholar API](https://www.semanticscholar.org/product/api)
2. **Get Key**: Follow their registration process
3. **Add to .env**: `SEMANTIC_SCHOLAR_API_KEY=your_key`

#### PubMed (Enhanced Medical Research)
1. **Register**: [NCBI API Key](https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/)
2. **Get Key**: Follow NCBI guidelines
3. **Add to .env**: `PUBMED_API_KEY=your_key`

## 🔧 Dependency Details

### Core Dependencies
```txt
PyQt5>=5.15.0          # GUI framework
openai>=1.0.0          # OpenAI API client
requests>=2.25.0       # HTTP requests
loguru>=0.6.0          # Logging
python-dotenv>=0.19.0  # Environment variables
asyncio-compat>=0.1.0  # Async support
```

### Optional Dependencies
```txt
transformers>=4.20.0   # Local AI models (optional)
torch>=1.12.0          # PyTorch for transformers
```

## 🐛 Troubleshooting Installation

### Common Issues

#### Python Version Error
```bash
# Check Python version
python3 --version

# If version < 3.8, update Python
sudo apt update && sudo apt install python3.9
```

#### PyQt5 Installation Issues
```bash
# Ubuntu/Debian
sudo apt-get install python3-pyqt5 python3-pyqt5-dev

# CentOS/RHEL
sudo yum install python3-qt5 python3-qt5-devel

# macOS
brew install pyqt5
```

#### Permission Errors
```bash
# Make scripts executable
chmod +x install.sh
chmod +x launch_gui.py

# If still issues, run with sudo (not recommended)
sudo python3 launch_gui.py
```

#### Virtual Environment Issues
```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Verification Steps

1. **Check Python Installation**
   ```bash
   python3 --version
   pip3 --version
   ```

2. **Verify Dependencies**
   ```bash
   python3 -c "import PyQt5; print('PyQt5 OK')"
   python3 -c "import openai; print('OpenAI OK')"
   ```

3. **Test Application**
   ```bash
   python3 -c "from rag_desktop_gui import main; print('GUI imports OK')"
   ```

## 🚦 Post-Installation

### First Run
1. Launch the application: `python3 launch_gui.py`
2. You should see the Klein blue and white interface
3. Try a test search: "machine learning"
4. Verify AI enhancement works

### Desktop Integration (Optional)
```bash
# Copy desktop file
cp RAG-Desktop-GUI.desktop ~/.local/share/applications/
```

### Performance Optimization
1. **For faster startup**: Keep virtual environment activated
2. **For GPU acceleration**: Install CUDA-enabled PyTorch (optional)
3. **For memory optimization**: Close other applications during large searches

## 📁 Directory Structure After Installation

```
MiniProjectEmerg/
├── .env                      # Your API keys (do not share)
├── launch_gui.py             # Application launcher
├── rag_desktop_gui.py        # Main application
├── requirements.txt          # Dependencies
├── venv/                     # Virtual environment (if used)
├── logs/                     # Application logs
├── config/                   # Configuration files
├── src/                      # Source code
└── dist/                     # Compiled executable
```

## 🔄 Updates

### Updating the Application
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

### Updating Dependencies
```bash
pip install --upgrade -r requirements.txt
```

## 🆘 Getting Help

If you encounter issues:

1. **Check this guide** for common solutions
2. **Review error messages** in the terminal
3. **Check logs** in the `logs/` directory
4. **Verify API keys** are correctly set
5. **Test internet connection** for API access

### Support Checklist
- [ ] Python 3.8+ installed
- [ ] All dependencies installed
- [ ] OpenAI API key set in .env
- [ ] Internet connection working
- [ ] No firewall blocking Python

---

**Installation complete! You're ready to start researching! 🔬** 