# 🔬 AI Research Assistant

A comprehensive, privacy-focused research assistant that uses local AI models to analyze academic papers and provide intelligent research insights. The system operates completely offline with local AI models, ensuring complete privacy and data security.

## ✨ Features

- **Local AI Models**: Runs entirely offline using local AI models for complete privacy
- **Intelligent Research Analysis**: Multi-section analysis with enhanced summarization and hypothesis generation
- **Academic Paper Integration**: Supports ArXiv, PubMed, and Semantic Scholar paper sources
- **Reinforcement Learning Optimization**: Continuously improves research quality through RL techniques
- **Enhanced RAG Pipeline**: Retrieval-Augmented Generation for contextual research insights
- **Modern GUI Interface**: User-friendly graphical interface for easy interaction
- **Database Management**: Local paper database with advanced search capabilities

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Git
- 4GB+ RAM recommended
- 10GB+ free disk space

### For Ubuntu Virtual Machine (VM)

1. **Update System Packages**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. **Install Required System Dependencies**
   ```bash
   sudo apt install -y python3 python3-pip python3-venv git wget curl
   sudo apt install -y build-essential libssl-dev libffi-dev python3-dev
   sudo apt install -y pkg-config libhdf5-dev
   ```

3. **Install Additional Dependencies for GUI**
   ```bash
   sudo apt install -y python3-tk python3-tkinter
   sudo apt install -y libgtk-3-dev libcairo2-dev libgirepository1.0-dev
   ```

4. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd MiniProjectEmerg
   ```

5. **Create and Activate Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

6. **Install Python Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

7. **Setup Local AI Models**
   ```bash
   python scripts/setup_local_database.py
   ```

### For Windows Terminal

1. **Install Python**
   - Download Python 3.8+ from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"
   - Verify installation: `python --version`

2. **Install Git**
   - Download Git from [git-scm.com](https://git-scm.com/downloads)
   - Install with default settings

3. **Open Windows Terminal/PowerShell as Administrator**

4. **Clone the Repository**
   ```powershell
   git clone <repository-url>
   cd MiniProjectEmerg
   ```

5. **Create and Activate Virtual Environment**
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

6. **Install Python Dependencies**
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

7. **Setup Local AI Models**
   ```powershell
   python scripts/setup_local_database.py
   ```

## 🚀 Running the Application

### Option 1: GUI Application (Recommended)
```bash
# Linux/Ubuntu
python launch_gui.py

# Windows
python launch_gui.py
```

### Option 2: Command Line Interface
```bash
# Linux/Ubuntu
python main.py

# Windows
python main.py
```

### Option 3: Direct GUI Access
```bash
# Linux/Ubuntu
python research_assistant_gui.py

# Windows
python research_assistant_gui.py
```

## 🔧 Environment Setup Details

### Virtual Environment Setup
After cloning the repository, users must:

1. **Create a virtual environment**
   ```bash
   # Linux/Ubuntu
   python3 -m venv venv
   
   # Windows
   python -m venv venv
   ```

2. **Activate the virtual environment**
   ```bash
   # Linux/Ubuntu
   source venv/bin/activate
   
   # Windows
   venv\Scripts\activate
   ```

3. **Install dependencies using requirements.txt**
   ```bash
   pip install -r requirements.txt
   ```

### Additional Dependencies

The system requires several additional components:

- **Local AI Models**: Automatically downloaded during first run
- **Database Setup**: Local SQLite database for paper storage
- **Tokenizer Components**: For text processing and embeddings
- **GUI Libraries**: Tkinter and supporting libraries for the interface

### Configuration

The system uses local-only configuration stored in `config/api_keys.py`:
- No external API keys required
- All processing happens locally
- Privacy-focused design with no data transmission

## 📁 Project Structure

```
MiniProjectEmerg/
├── config/                 # Configuration files
├── data/                   # Database and model storage
│   ├── models/            # Local AI models
│   ├── papers/            # Downloaded papers
│   └── cache/             # Cached results
├── src/                    # Source code
│   ├── api/               # API clients
│   ├── database/          # Database management
│   ├── models/            # AI model implementations
│   ├── rag/               # RAG pipeline
│   └── rl/                # Reinforcement learning
├── scripts/               # utility scripts
├── logs/                  # Application logs
├── venv/                  # Virtual environment
├── requirements.txt       # Python dependencies
├── launch_gui.py         # GUI launcher
└── main.py               # CLI entry point
```

## 🎯 Usage

1. **Start the application** using one of the methods above
2. **Enter your research query** in the interface
3. **Wait for analysis** - the system will:
   - Search local database for relevant papers
   - Generate enhanced summaries
   - Provide research hypotheses
   - Suggest future research directions
4. **Review results** in the comprehensive output sections

## 🏆 Team Members

- **Mark Daniel Ortiz** - Lead Developer & System Architect
- **Jan Adrian Manzanero** - AI/ML Engineer & Model Optimization Specialist
- **Neil Emmanuel Macaro** - Database Engineer & Backend Developer
- **Vinz Bequilla** - UI/UX Designer & Frontend Developer

## 📋 System Requirements

- **Operating System**: Ubuntu 18.04+ or Windows 10+
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 10GB free space for models and papers
- **Network**: Internet connection for initial setup and paper downloads

## 🔒 Privacy & Security

- **100% Local Processing**: All AI computations happen locally
- **No Data Transmission**: Research data never leaves your machine
- **Offline Capable**: Works completely offline after initial setup
- **Privacy First**: No external API calls for AI processing

## 🐛 Troubleshooting

### Common Issues

1. **Module Import Errors**: Ensure virtual environment is activated
2. **GUI Not Starting**: Install tkinter packages for your system
3. **Model Loading Issues**: Check available disk space and memory
4. **Database Errors**: Run `python scripts/setup_local_database.py`

### Getting Help

If you encounter issues:
1. Check the `logs/` directory for error messages
2. Ensure all dependencies are installed correctly
3. Verify Python version compatibility
4. Try running in a fresh virtual environment

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Built with ❤️ for privacy-focused AI research* 