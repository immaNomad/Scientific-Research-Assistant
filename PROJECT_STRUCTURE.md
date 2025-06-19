# Project Structure Guide

This document explains the clean, organized structure of the Scientific Research Assistant project after cleanup.

## 📁 Root Directory Layout

```
MiniProjectEmerg/
├── 📱 Application Files
│   ├── launch_gui.py              # Application launcher - START HERE
│   ├── rag_desktop_gui.py         # Main GUI application (PyQt5)
│   └── main.py                    # Command-line interface
│
├── 📚 Documentation
│   ├── README.md                  # Main user guide and features
│   ├── INSTALLATION.md            # Setup and installation guide
│   ├── DEVELOPER.md              # Developer documentation
│   └── PROJECT_STRUCTURE.md      # This file
│
├── ⚙️ Configuration
│   ├── requirements.txt           # Python dependencies
│   ├── env.example               # Environment variables template
│   ├── install.sh                # Automated installation script
│   └── RAG-Desktop-GUI.desktop   # Linux desktop integration
│
├── 🧠 Core Source Code
│   ├── config/                   # Application configuration
│   │   └── config.py            # Settings and parameters
│   └── src/                     # Main source code modules
│       ├── models/              # AI/ML model integrations
│       ├── rag/                 # RAG pipeline implementation
│       ├── api/                 # External API clients
│       └── utils/               # Utility functions
│
├── 📦 Distribution & Runtime
│   ├── dist/                    # Compiled executables
│   │   └── RAG-Research-Assistant
│   ├── data/                    # Application data storage
│   ├── logs/                    # Application logs
│   └── venv/                    # Python virtual environment (optional)
```

## 🎯 Entry Points

### Primary Entry Point
```bash
python3 launch_gui.py
```
**Use Case**: Normal users who want the desktop GUI application

### Alternative Entry Points
```bash
# Command-line interface
python3 main.py

# Direct GUI launch
python3 rag_desktop_gui.py
```

## 📋 Essential Files

### For Users
| File | Purpose | Required |
|------|---------|----------|
| `launch_gui.py` | Application launcher | ✅ Yes |
| `rag_desktop_gui.py` | Main GUI application | ✅ Yes |
| `README.md` | User guide and features | ✅ Yes |
| `INSTALLATION.md` | Setup instructions | ✅ Yes |
| `requirements.txt` | Dependencies list | ✅ Yes |
| `env.example` | API key template | ✅ Yes |
| `install.sh` | Installation script | ✅ Yes |

### For Developers
| File | Purpose | Required |
|------|---------|----------|
| `DEVELOPER.md` | Code documentation | 🔧 Development |
| `PROJECT_STRUCTURE.md` | This guide | 📖 Reference |
| `main.py` | CLI interface | 🔧 Development |
| `config/config.py` | App configuration | ⚙️ Configuration |
| `src/` directory | Source code modules | 🧠 Core logic |

### Runtime Generated
| Directory | Purpose | Auto-Created |
|-----------|---------|--------------|
| `logs/` | Application logs | ✅ Runtime |
| `data/` | Application data | ✅ Runtime |
| `venv/` | Virtual environment | 🔧 Optional |
| `dist/` | Compiled executables | 📦 Build time |

## 🚀 Quick Start Flow

### For End Users
1. **Read**: `README.md` for overview
2. **Install**: Follow `INSTALLATION.md`
3. **Configure**: Copy `env.example` to `.env` and add API key
4. **Launch**: Run `python3 launch_gui.py`

### For Developers
1. **Read**: `DEVELOPER.md` for architecture
2. **Setup**: Follow installation guide
3. **Explore**: Study `rag_desktop_gui.py` for GUI logic
4. **Extend**: Modify code in `src/` directories

## 🧹 What Was Removed

During cleanup, the following unnecessary files were removed:
- `web_gui.py` - Web interface (not needed for desktop app)
- `demo.py` - Demo scripts
- `test_llm.py` - Test files
- `build_executable.py` - Redundant build script
- `setup.py` - Complex setup (replaced with simple install.sh)
- `USAGE.md` - Outdated usage guide
- Various documentation files - Consolidated into comprehensive guides
- Empty directories (`docs/`, `tests/`)

## 🔧 Configuration Files

### Environment Variables (`.env`)
```bash
# Required for AI features
OPENAI_API_KEY=your_key_here

# Optional for enhanced features
SEMANTIC_SCHOLAR_API_KEY=optional_key
PUBMED_API_KEY=optional_key
```

### Application Config (`config/config.py`)
- API rate limits
- Model parameters
- Search configurations
- File paths and directories

## 📊 Directory Size Guide

| Directory | Typical Size | Contents |
|-----------|--------------|----------|
| Root files | ~50MB | Python code and docs |
| `src/` | ~10-20MB | Source code modules |
| `dist/` | ~100-200MB | Compiled executable |
| `venv/` | ~200-500MB | Virtual environment |
| `logs/` | ~1-10MB | Application logs |
| `data/` | ~1-50MB | Cached search results |

## 🔍 Finding What You Need

### "I want to use the application"
➡️ Start with `README.md` then `INSTALLATION.md`

### "I want to understand the code"
➡️ Read `DEVELOPER.md` then explore `rag_desktop_gui.py`

### "I want to modify features"
➡️ Study `DEVELOPER.md` and source files in `src/`

### "I want to add new functionality"
➡️ Check extension points in `DEVELOPER.md`

### "I'm having problems"
➡️ Check troubleshooting in `README.md` and `INSTALLATION.md`

## 🧭 Navigation Tips

### Code Organization
- **GUI Logic**: `rag_desktop_gui.py` (main file)
- **CLI Logic**: `main.py`
- **Configuration**: `config/config.py`
- **AI Integration**: `src/models/`
- **Search Logic**: `src/rag/` and `src/api/`

### Documentation Hierarchy
1. **README.md** - Overview and features
2. **INSTALLATION.md** - Setup guide
3. **DEVELOPER.md** - Technical details
4. **PROJECT_STRUCTURE.md** - Organization (this file)

## 📝 File Naming Conventions

- **Snake case**: Python files (`rag_desktop_gui.py`)
- **Kebab case**: Documentation (`PROJECT_STRUCTURE.md`)
- **UPPERCASE**: Important files (`README.md`, `INSTALLATION.md`)
- **Descriptive names**: Clear purpose indication

---

**This structure makes the codebase clean, maintainable, and easy to understand! 🎯** 