# Scientific Research Assistant

A powerful desktop application for scientific literature search and analysis with AI-powered insights generation.

## 🌟 Features

- **🔍 Literature Search**: Search across arXiv, Semantic Scholar, and PubMed databases
- **🧠 AI-Powered Analysis**: Generate comprehensive abstracts and research hypotheses using OpenAI
- **📊 Smart Results Display**: Truncated view with "Show More" functionality for better UX
- **🎨 Modern UI**: Clean Klein blue and white interface
- **⚡ Fast Performance**: Multi-threaded search with progress indication

## 📋 Requirements

- Python 3.8+
- OpenAI API Key (for AI analysis)
- Internet connection

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd MiniProjectEmerg
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure OpenAI API
Create a `.env` file in the project root:
```bash
cp env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

### 4. Launch Application
```bash
python3 launch_gui.py
```

## 📖 How to Use

### Basic Search
1. **Enter Query**: Type your research question in the text box
2. **Click Search**: Use "🔍 Search Papers" for specific queries
3. **View Results**: Results appear with AI-generated abstracts and hypotheses
4. **Show More**: Click "📄 Show More Details" to see full abstracts

### Search Options
- **🔍 Search Papers**: Search for specific research topics
- **📄 Recent Papers**: Get latest papers from arXiv
- **🔬 Full Analysis**: Complete analysis with enhanced AI insights
- **🗑️ Clear**: Reset all fields

### Understanding Results

The application provides three types of analysis:

#### 1. Comprehensive Abstract
- 300-500 word synthesis of all found papers
- Key findings, methodologies, and trends
- Research gaps and implications

#### 2. Research Hypotheses
- 3-5 testable research hypotheses
- Based on identified literature gaps
- Novel and measurable proposals

#### 3. Original Search Results
- Raw paper listings with abstracts
- Author information and publication details
- Direct links to papers

## 📁 Project Structure

```
MiniProjectEmerg/
├── launch_gui.py              # Application launcher
├── rag_desktop_gui.py         # Main GUI application
├── main.py                    # Command-line interface
├── requirements.txt           # Python dependencies
├── env.example               # Environment variables template
├── install.sh                # Installation script
├── config/
│   └── config.py             # Application configuration
├── src/
│   ├── models/               # LLM and embedding models
│   ├── rag/                  # RAG pipeline implementation
│   ├── api/                  # API clients for literature sources
│   └── utils/                # Utility functions
├── dist/
│   └── RAG-Research-Assistant # Compiled executable
└── logs/                     # Application logs
```

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key for AI analysis
- `SEMANTIC_SCHOLAR_API_KEY`: Optional for higher rate limits
- `PUBMED_API_KEY`: Optional for PubMed access

### Application Settings
Edit `config/config.py` to customize:
- Search result limits
- API rate limits
- Model configurations
- Database settings

## 🎨 User Interface

### Klein Blue Theme
- **Primary Color**: Klein Blue (#002FA7)
- **Background**: White (#FFFFFF)
- **Text**: Black (#000000)
- **Accents**: Various blue shades

### Responsive Design
- Minimum window size: 1200x800
- Resizable interface
- Scrollable results area
- Progress indicators

## 🔍 Search Sources

### arXiv
- Preprint repository for physics, mathematics, computer science
- Real-time access to latest research
- No API key required

### Semantic Scholar
- Academic paper search with citation analysis
- Broad coverage across disciplines
- Optional API key for enhanced access

### PubMed
- Biomedical and life science literature
- Comprehensive medical research database
- API key recommended for full access

## 🤖 AI Integration

### OpenAI Features
- GPT-3.5-turbo or GPT-4 models supported
- Comprehensive abstract generation
- Novel hypothesis creation
- Contextual analysis of research trends

### Fallback Support
- Template-based analysis if OpenAI unavailable
- Basic summary generation
- Maintains functionality without API

## 🛠️ Development

### Running from Source
```bash
python3 rag_desktop_gui.py
```

### Using Command Line
```bash
python3 main.py search "machine learning"
python3 main.py recent cs.AI 7
python3 main.py full "neural networks"
```

### Building Executable
```bash
./install.sh
```

## 🐛 Troubleshooting

### Common Issues

#### "No papers found"
- Check internet connection
- Verify search terms are not too specific
- Try different search sources

#### "LLM processing failed"
- Verify OpenAI API key is set correctly
- Check API quota and billing
- Application will fallback to template-based analysis

#### "Executable not found"
- Run `./install.sh` to build executable
- Ensure dist/RAG-Research-Assistant exists
- Check file permissions

### Error Messages
- **Transform warnings**: CSS styling warnings (safe to ignore)
- **Transformers not available**: Local AI models disabled (OpenAI still works)
- **PyQt5 required**: Install with `pip install PyQt5`

## 📄 License

This project is licensed under the MIT License.

## 👥 Contributors

Group 4 - Mapúa University

## 🆘 Support

For issues and questions:
1. Check this documentation
2. Review error messages in terminal
3. Verify configuration settings
4. Check logs in `logs/` directory

---

**Happy Researching! 🔬** 