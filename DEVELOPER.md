# Developer Documentation

This document provides detailed information about the Scientific Research Assistant codebase for developers who want to understand, modify, or extend the application.

## 🏗️ Architecture Overview

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GUI Layer     │    │  Command Line   │    │   Desktop App   │
│ (rag_desktop_   │    │   Interface     │    │   Launcher      │
│    gui.py)      │    │   (main.py)     │    │ (launch_gui.py) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────────────────────────────────────┐
         │               Core Components                    │
         └─────────────────────────────────────────────────┘
                                 │
    ┌────────────────────────────┼────────────────────────────┐
    │                            │                            │
┌─────────────┐        ┌─────────────┐              ┌─────────────┐
│ RAG Pipeline│        │ LLM Manager │              │ API Clients │
│  (src/rag)  │        │(src/models) │              │  (src/api)  │
└─────────────┘        └─────────────┘              └─────────────┘
    │                            │                            │
    │                   ┌─────────────┐                      │
    └───────────────────│ Config &    │──────────────────────┘
                        │ Utils       │
                        │(config/src) │
                        └─────────────┘
```

### Key Components

1. **GUI Layer** (`rag_desktop_gui.py`): PyQt5-based desktop interface
2. **CLI Layer** (`main.py`): Command-line interface for batch processing
3. **RAG Pipeline** (`src/rag/`): Core literature search and analysis logic
4. **LLM Integration** (`src/models/`): AI-powered text generation
5. **API Clients** (`src/api/`): External service integrations
6. **Configuration** (`config/`): Application settings and environment

## 📁 Detailed File Structure

```
MiniProjectEmerg/
├── 📱 User Interfaces
│   ├── launch_gui.py           # GUI launcher and entry point
│   ├── rag_desktop_gui.py      # Main PyQt5 desktop application
│   └── main.py                 # Command-line interface
│
├── ⚙️ Configuration
│   ├── config/
│   │   └── config.py           # Application configuration settings
│   ├── env.example             # Environment variables template
│   └── requirements.txt        # Python dependencies
│
├── 🧠 Core Components
│   └── src/
│       ├── models/             # AI and ML models
│       │   ├── llm_client.py   # LLM provider integrations
│       │   └── embeddings.py   # Text embedding models
│       ├── rag/                # RAG pipeline implementation
│       │   ├── rag_pipeline.py # Main RAG orchestration
│       │   └── vector_store.py # Document storage and retrieval
│       ├── api/                # External API clients
│       │   ├── arxiv_client.py # arXiv API integration
│       │   ├── semantic_client.py # Semantic Scholar API
│       │   └── pubmed_client.py   # PubMed API
│       └── utils/              # Utility functions
│           ├── text_utils.py   # Text processing utilities
│           └── logging_utils.py # Logging configuration
│
├── 📦 Distribution
│   ├── dist/                   # Compiled executables
│   │   └── RAG-Research-Assistant
│   └── install.sh              # Installation script
│
├── 📊 Data & Logs
│   ├── data/                   # Application data storage
│   └── logs/                   # Application logs
│
└── 📚 Documentation
    ├── README.md               # Main documentation
    ├── INSTALLATION.md         # Installation guide
    └── DEVELOPER.md           # This developer guide
```

## 🎨 GUI Architecture (`rag_desktop_gui.py`)

### Class Structure

```python
class RAGDesktopGUI(QMainWindow):
    """Main application window"""
    
    # UI Creation Methods
    def init_ui(self)              # Initialize the user interface
    def create_header(self)        # Create header section (empty in current version)
    def create_input_section(self) # Create query input and controls
    def create_results_section(self) # Create results display area
    
    # Search Methods
    def search_papers(self)        # Handle search button click
    def recent_papers(self)        # Handle recent papers button
    def full_analysis(self)        # Handle full analysis button
    def execute_search(self)       # Core search execution logic
    
    # Result Processing
    def on_search_finished(self)   # Handle search completion
    def generate_enhanced_analysis(self) # LLM-powered result enhancement
    def create_truncated_results(self)   # Create truncated display version
    def show_more_details(self)    # Toggle between full/truncated view
    
    # Utility Methods
    def build_command(self)        # Build executable command
    def get_sources(self)          # Get selected data sources
    def clear_all(self)           # Reset application state
```

### Threading Model

```python
class SearchWorker(QThread):
    """Background thread for search operations"""
    
    # Signals
    finished = pyqtSignal(bool, str, str)  # success, output, error
    progress = pyqtSignal(str)             # status updates
    
    # Methods
    def run(self)  # Execute search command in background
```

### UI Components

1. **Query Input Section**
   - Text area for research queries
   - Placeholder text with examples
   - Input validation

2. **Action Buttons**
   - Search Papers: Targeted literature search
   - Recent Papers: Latest publications
   - Full Analysis: Complete AI analysis
   - Clear: Reset interface

3. **Results Display**
   - Truncated view by default
   - "Show More" toggle functionality
   - Formatted text with syntax highlighting

4. **Status Management**
   - Progress indication during searches
   - Error handling and user feedback
   - Status bar updates

## 🔧 Configuration System

### Environment Variables (`.env`)

```bash
# Required
OPENAI_API_KEY=sk-your-openai-key-here

# Optional
SEMANTIC_SCHOLAR_API_KEY=your-semantic-scholar-key
PUBMED_API_KEY=your-pubmed-key

# Debug Settings
DEBUG=false
LOG_LEVEL=INFO
```

### Application Config (`config/config.py`)

```python
class Config:
    """Application configuration"""
    
    # API Settings
    api = APIConfig()
    
    # Model Settings  
    model = ModelConfig()
    
    # RAG Pipeline Settings
    rag = RAGConfig()
    
    # Database Settings
    database = DatabaseConfig()
```

Key configuration areas:
- **API Rate Limits**: Control request frequency
- **Model Parameters**: LLM and embedding settings
- **Search Limits**: Maximum results per query
- **File Paths**: Data and log directory locations

## 🧠 LLM Integration

### LLM Manager Architecture

```python
class LLMManager:
    """Manages multiple LLM providers"""
    
    def __init__(self):
        self.clients = {
            'openai': OpenAIClient(),
            'local': LocalLLMClient(),
            'fallback': FallbackLLMClient()
        }
    
    async def generate(self, prompt, max_tokens, temperature):
        """Generate text using available LLM"""
        # Priority: OpenAI -> Local -> Fallback
```

### LLM Providers

1. **OpenAI Client** (`OpenAIClient`)
   - GPT-3.5-turbo/GPT-4 integration
   - Async API calls
   - Cost and token tracking
   - Error handling and retries

2. **Local Client** (`LocalLLMClient`)
   - Transformers-based models
   - GPU acceleration support
   - Offline operation capability

3. **Fallback Client** (`FallbackLLMClient`)
   - Template-based generation
   - Always available
   - No API dependencies

### Enhancement Pipeline

```python
def generate_enhanced_analysis(self, raw_output):
    """Generate AI-enhanced analysis"""
    
    # 1. Extract query context
    query = self.query_text.toPlainText().strip()
    
    # 2. Create enhancement prompt
    enhancement_prompt = f"""
    Based on the following research query and search results,
    provide comprehensive analysis...
    """
    
    # 3. Generate enhanced content
    response = asyncio.run(llm_manager.generate(
        enhancement_prompt, 
        max_tokens=800, 
        temperature=0.7
    ))
    
    # 4. Format and return results
    return response.content + "\n\n" + raw_output
```

## 🔍 Search Pipeline

### Command Building

```python
def build_command(self, command_type, query=None):
    """Build executable command for search"""
    
    cmd = [str(self.exe_path), command_type]
    
    if query:
        cmd.append(query)
    
    sources = self.get_sources()  # ['arxiv', 'semantic_scholar', 'pubmed']
    if sources:
        cmd.extend(["--sources"] + sources)
    
    return cmd
```

### Search Execution Flow

1. **Validation**: Check inputs and executable availability
2. **Command Building**: Construct CLI command with parameters
3. **Background Execution**: Run search in separate thread
4. **Result Processing**: Parse and format search results
5. **LLM Enhancement**: Generate AI analysis
6. **Display**: Show results with truncation options

## 🎨 UI Styling and Themes

### Klein Blue Theme

```python
# Primary Colors
KLEIN_BLUE = "#002FA7"      # Main brand color
WHITE = "#ffffff"           # Background
BLACK = "#000000"           # Text color
LIGHT_BLUE = "#f8f9ff"      # Focus backgrounds
ACCENT_BLUE = "#0033CC"     # Hover states
```

### Component Styling

```python
# Button Styling
QPushButton {
    background-color: #002FA7;
    color: #ffffff;
    border: none;
    padding: 12px;
    border-radius: 6px;
    font-weight: bold;
}

# Input Field Styling  
QTextEdit {
    background-color: #ffffff;
    border: 2px solid #002FA7;
    border-radius: 6px;
    padding: 10px;
    color: #000000;
}
```

## 🔧 Extension Points

### Adding New Search Sources

1. **Create API Client** in `src/api/`
2. **Update Source List** in `get_sources()`
3. **Modify Command Building** to include new source
4. **Update Configuration** for API settings

### Custom LLM Providers

1. **Inherit from `BaseLLMClient`**
2. **Implement required methods**: `generate()`, `is_available()`
3. **Register in `LLMManager`**
4. **Add configuration options**

### UI Customization

1. **Theme Modification**: Update stylesheet in `init_ui()`
2. **New Components**: Add to appropriate `create_*_section()` method
3. **Layout Changes**: Modify layout managers and spacing
4. **Custom Widgets**: Create new PyQt5 widget classes

## 🧪 Testing and Debugging

### Debug Mode

Enable debug output:
```python
# Temporary debug prints
print(f"DEBUG: Command = {cmd}")
print(f"DEBUG: Output length = {len(output)}")
```

### Error Handling

```python
try:
    # LLM generation
    response = asyncio.run(llm_manager.generate(prompt))
    enhanced_content = response.content
except Exception as llm_error:
    # Graceful fallback
    return f"=== ENHANCED ANALYSIS UNAVAILABLE ===\n{raw_output}"
```

### Testing Guidelines

1. **Unit Tests**: Test individual methods
2. **Integration Tests**: Test API integrations
3. **UI Tests**: Test user interactions
4. **Error Scenarios**: Test failure modes

## 📊 Performance Considerations

### Search Optimization

- **Threading**: Background search execution
- **Caching**: Store recent results
- **Rate Limiting**: Respect API limits
- **Timeout Handling**: 5-minute search timeout

### Memory Management

- **Result Truncation**: Limit displayed content
- **Lazy Loading**: Load full results on demand
- **Garbage Collection**: Clean up after searches

### UI Responsiveness

- **Async Operations**: Non-blocking UI updates
- **Progress Indicators**: User feedback during operations
- **Error Recovery**: Graceful failure handling

## 🔒 Security Considerations

### API Key Management

- **Environment Variables**: Never hardcode keys
- **File Permissions**: Secure .env file access
- **Version Control**: Exclude sensitive files

### Input Validation

- **Query Sanitization**: Clean user inputs
- **Command Injection**: Prevent malicious commands
- **Error Information**: Limit exposed error details

## 🚀 Deployment

### Building Executable

```bash
# Use provided build script
./install.sh

# Manual build with PyInstaller
pyinstaller --onefile --windowed rag_desktop_gui.py
```

### Distribution Package

Include in release:
- Executable binary (`dist/RAG-Research-Assistant`)
- Configuration files (`config/`, `env.example`)
- Documentation (`README.md`, `INSTALLATION.md`)
- Installation script (`install.sh`)

---

**Happy Developing! 🚀** 