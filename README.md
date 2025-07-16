# 🚀 AI Research Assistant

A privacy-focused, locally-running research assistant that helps you search, analyze, and generate insights from AI/ML research papers using custom trained models.

## ✨ Features

- 🔍 **Advanced Paper Search** - Topic-focused search across 146+ research papers
- 🤖 **Custom AI Models** - Locally trained models, no external APIs required
- 📚 **Literature Analysis** - Automated paper summarization and insights
- 💡 **Hypothesis Generation** - AI-powered research hypothesis creation
- 🔄 **Reinforcement Learning** - Self-improving search optimization
- 📊 **Performance Analytics** - Query optimization and system monitoring
- 🔒 **Complete Privacy** - All processing happens locally, no data leaves your machine

## 🎯 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ai-research-assistant.git
cd ai-research-assistant
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup the System
```bash
# Create required directories
mkdir -p data/papers data/models/local_ai data/cache data/embeddings data/feedback logs

# Setup configuration
cp config/api_keys.example.py config/api_keys.py

# Populate database with research papers
python scripts/populate_database.py

# Train local AI models
python scripts/train_local_model.py
```

### 4. Launch the Application
```bash
# GUI Application
python launch_gui.py

# Command Line Interface
python main.py
```

## 🔧 Alternative Setup Options

### Individual Step Commands
If you prefer to run each step separately:

```bash
# Step 1: Create directories
mkdir -p data/papers data/models/local_ai data/cache data/embeddings data/feedback logs

# Step 2: Setup configuration
cp config/api_keys.example.py config/api_keys.py

# Step 3: Populate database (takes 10-15 minutes)
python scripts/populate_database.py

# Step 4: Train models (takes 20-30 minutes)
python scripts/train_local_model.py
```

## 📊 System Requirements

- **Python**: 3.8+
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ free space
- **CPU**: Multi-core recommended for training
- **OS**: Linux, macOS, Windows

## 🔒 Privacy & Security

This system is designed with privacy as a core principle:

- ✅ **No External APIs** - All AI processing happens locally
- ✅ **No Data Collection** - Your research stays on your machine
- ✅ **Open Source** - Full transparency, inspect all code
- ✅ **Offline Capable** - Works without internet connection
- ✅ **Custom Models** - Train your own AI models on your data

## 🎮 Usage Examples

### GUI Application
1. Launch: `python launch_gui.py`
2. Enter research query (e.g., "transformer attention mechanisms")
3. View results with AI-generated insights and hypotheses

### Command Line
```bash
# Search papers
python main.py search "machine learning optimization"

# Full analysis with hypothesis generation
python main.py full "neural network architectures"

# Interactive mode
python main.py interactive
```

## 📁 Project Structure

```
ai-research-assistant/
├── src/
│   ├── database/          # Paper database and search engines
│   ├── models/            # AI models and embeddings
│   ├── rag/               # RAG pipelines and analysis
│   ├── rl/                # Reinforcement learning optimization
│   └── analytics/         # Performance monitoring
├── scripts/               # Setup and utility scripts
├── config/                # Configuration files
├── data/                  # Generated data (not in git)
│   ├── papers/           # Research paper database
│   ├── models/           # Trained AI models
│   └── cache/            # Search optimization cache
└── logs/                  # Application logs
```

## 🧪 Advanced Features

### Custom Model Training
```bash
python scripts/train_local_model.py --epochs 50 --batch_size 16
```

### Database Management
```bash
python scripts/populate_database.py --target_papers 200
```

### Performance Optimization
```bash
python optimize_system.py
```

### System Monitoring
```bash
python scripts/monitor_performance.py
```

## 🛠️ Troubleshooting

### Common Issues

1. **"No module named 'transformers'"**
   ```bash
   pip install -r requirements.txt
   ```

2. **"Database not found"**
   ```bash
   python scripts/populate_database.py
   ```

3. **"Models not loading"**
   ```bash
   python scripts/train_local_model.py
   ```

4. **Memory issues during training**
   - Reduce batch size in training scripts
   - Use CPU instead of GPU if needed

### System Check
```bash
python -c "
import sys; sys.path.append('src')
from database.paper_db import PaperDatabase
print(f'Papers: {len(PaperDatabase().get_all_papers())}')
"
```

## 📚 Research Paper Sources

The system includes papers from:
- arXiv (AI/ML/CS categories)
- Semantic Scholar
- Focus areas: Deep Learning, Computer Vision, NLP, Cybersecurity

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🐛 Bug Reports

Please report bugs by creating an issue with:
- System information (OS, Python version)
- Error messages and logs
- Steps to reproduce

## 👥 Team Members

- **Mark Daniel Ortiz** - Lead Developer & System Architect
- **Jan Adrian Manzanero** - AI/ML Engineer & Model Optimization Specialist  
- **Neil Emmanuel Macaro** - Database Engineer & Backend Developer
- **Vinz Bequilla** - UI/UX Designer & Frontend Developer

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for transformer models
- arXiv for open research papers
- The open-source AI community

## 📞 Support
- 📧 Email: mdsortiz@mymail.mapua.edu.ph
- 🐛 Issues: [GitHub Issues](https://github.com/immaNomad/Scientific-Research-Assistant/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/immaNomad/Scientific-research-assistant/discussions)
---

**Made with ❤️ for researchers, by researchers** 
