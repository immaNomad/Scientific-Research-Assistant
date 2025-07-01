# 🧠 AI Research Assistant - School Project

An intelligent research assistant that uses AI to help you find and analyze academic papers. Built with Python, this tool combines Retrieval-Augmented Generation (RAG) with Reinforcement Learning to provide smart research capabilities.

## ✨ Features

- 🔍 **Smart Paper Search**: Search across multiple academic databases (arXiv, Semantic Scholar)
- 🧠 **AI Analysis**: Powered by Google Gemini AI for intelligent summaries
- 📊 **Interactive GUI**: Easy-to-use desktop interface
- 🎯 **Learning System**: Uses reinforcement learning to improve search results
- 📝 **Research Reports**: Generate comprehensive analysis reports

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone or download the project
cd MiniProjectEmerg

# Activate the virtual environment
source venv/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
python launch_gui.py
```

That's it! The application is ready to use with a pre-configured API key.

## 💡 How to Use

1. **Launch the GUI** using the command above
2. **Type your research question** in the input field
3. **Click "Search"** to find relevant papers
4. **Review the AI analysis** and paper recommendations
5. **Export results** if needed

## 📁 Project Structure

```
MiniProjectEmerg/
├── src/                    # Core application code
│   ├── api/               # API clients for paper databases
│   ├── models/            # AI and embedding models
│   ├── rag/               # RAG pipeline implementation
│   └── rl/                # Reinforcement learning optimizer
├── config/                # Configuration files (API keys included)
├── data/                  # Data storage (cache, papers, etc.)
├── launch_gui.py          # Main launcher script
└── requirements.txt       # Python dependencies
```

## 🎓 Educational Purpose

This project demonstrates:
- **Information Retrieval**: Searching academic databases
- **Natural Language Processing**: AI-powered text analysis
- **Machine Learning**: Reinforcement learning for optimization
- **Software Engineering**: Modular Python application design
- **GUI Development**: Desktop application with Tkinter

## 📋 Requirements

- Python 3.11+
- Internet connection (for API calls)
- Virtual environment (included)

## 🔧 Technical Details

- **Frontend**: Tkinter GUI
- **AI Model**: Google Gemini (via API)
- **Search Sources**: arXiv, Semantic Scholar
- **Learning**: Q-learning reinforcement learning
- **Architecture**: Modular Python with RAG pipeline

## 📖 Example Queries

Try asking questions like:
- "Latest advances in machine learning"
- "Cybersecurity threats in 2024"
- "Climate change research"
- "Medical AI applications"

---

**Note**: This is an educational project with included API keys for demonstration purposes. For production use, obtain your own API keys from the respective services. 