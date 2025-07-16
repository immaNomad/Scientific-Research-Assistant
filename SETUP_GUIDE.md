# 🚀 Research Assistant Setup Guide

This guide will help you set up the AI Research Assistant with all required data files and models.

## 📋 Prerequisites

- Python 3.8+
- Git
- 2GB+ free disk space
- Internet connection (for initial setup)

## 🔧 Quick Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/research-assistant.git
cd research-assistant
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Setup Script
```bash
python setup.py
```

This will automatically:
- Create required directories
- Download/collect research papers
- Set up the database
- Train local AI models
- Initialize configuration files

### 4. Launch the Application
```bash
python launch_gui.py
```

## 📁 Manual Setup (if needed)

### Required Data Structure:
```
data/
├── papers/
│   ├── papers.db           # SQLite database (will be created)
│   └── [paper_dirs]/       # Individual paper directories
├── models/
│   └── local_ai/
│       ├── best_model.pt   # Trained AI model
│       ├── tokenizer/      # Tokenizer files
│       └── *.json         # Model configs
├── rl_experience.json      # RL training data
└── analytics.db           # Analytics database
```

### Step-by-Step Manual Setup:

#### 1. Create Data Directories
```bash
mkdir -p data/papers
mkdir -p data/models/local_ai
mkdir -p data/cache
mkdir -p data/embeddings
mkdir -p data/feedback
mkdir -p logs
```

#### 2. Populate Database with Papers
```bash
python scripts/populate_database.py
```
This will:
- Download 146 AI/ML/Cybersecurity papers
- Create SQLite database
- Extract metadata and abstracts
- Set up search indexing

#### 3. Train Local AI Models
```bash
python scripts/train_local_model.py
```
This will:
- Train custom AI model on collected papers
- Create tokenizer and model files
- Generate model configuration files

#### 4. Initialize Configuration
```bash
cp config/api_keys.example.py config/api_keys.py
```

## 🔒 Privacy & Local Mode

The system runs in **LOCAL MODELS ONLY** mode by default:
- ✅ No external API calls
- ✅ Complete privacy
- ✅ Offline functionality
- ✅ Custom trained models

## 📊 System Requirements

| Component | Size | Purpose |
|-----------|------|---------|
| Database | ~53MB | 146 research papers |
| AI Models | ~522MB | Custom trained models |
| Cache | Variable | Search optimization |
| Logs | Variable | System monitoring |

## 🎯 Features Available

- 🔍 Advanced paper search
- 📚 Literature analysis
- 💡 Hypothesis generation
- 🤖 AI-powered insights
- 🔄 Reinforcement learning optimization
- 📊 Performance analytics

## 🛠️ Troubleshooting

### Common Issues:

1. **Database not found**
   ```bash
   python scripts/setup_local_database.py
   ```

2. **Models not loading**
   ```bash
   python scripts/train_local_model.py
   ```

3. **Import errors**
   ```bash
   pip install -r requirements.txt
   ```

### System Check:
```bash
python -c "
import sys
sys.path.append('src')
from database.paper_db import PaperDatabase
db = PaperDatabase()
print(f'Database: {db.count_papers()} papers')
"
```

## 📞 Support

If you encounter issues:
1. Check the logs in `logs/` directory
2. Run the setup script again
3. Ensure all dependencies are installed
4. Verify data directories exist

## 🚀 Advanced Usage

### Custom Model Training:
```bash
python scripts/train_local_model.py --epochs 50 --batch_size 16
```

### Database Management:
```bash
python scripts/populate_database.py --target_papers 200
```

### Performance Optimization:
```bash
python optimize_system.py
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details. 