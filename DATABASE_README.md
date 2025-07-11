# Local AI/ML Paper Database System

## Overview

This project has been transformed from using external APIs (arXiv, Semantic Scholar, PubMed) to a **local database system** that stores and manages AI/ML research papers. The system now:

1. **Downloads and stores 50+ AI/ML papers** in a local SQLite database
2. **Stores PDFs in separate folders** for each paper  
3. **Uses only arXiv and Semantic Scholar** as sources
4. **Provides enhanced content generation** through local RAG system
5. **Hides the dataset** using RAG so users don't have direct access

## 🎯 Key Features

### Local Database System
- **SQLite database** for efficient paper storage and retrieval
- **Individual PDF folders** for each paper (organized by paper ID)
- **Metadata management** including titles, authors, abstracts, citations
- **Source tracking** (arXiv vs Semantic Scholar)

### Enhanced Content Generation
- **Local RAG pipeline** for intelligent paper retrieval
- **LLM-powered summarization** of research findings
- **Hypothesis generation** based on local paper collection
- **Domain-aware search** routing (AI/ML, healthcare, physics, etc.)

### Privacy & Control
- **No external API dependencies** during operation
- **Complete data control** - all papers stored locally
- **Offline capability** once database is populated
- **Hidden dataset access** - users interact through RAG, not direct database queries

## 🏗️ Architecture

```
src/
├── database/
│   ├── paper_db.py           # SQLite database management
│   ├── paper_collector.py    # Downloads papers from arXiv/Semantic Scholar
│   └── local_search.py       # Local search engine replacing APIs
├── rag/
│   ├── local_rag_pipeline.py      # Local RAG system
│   └── local_enhanced_rag.py      # Enhanced content generation
└── rl/
    └── rl_optimizer.py       # RL system adapted for local database

data/
└── papers/
    ├── papers.db            # SQLite database
    ├── [paper_id_1]/        # Individual paper folders
    │   └── [arxiv_id].pdf   # PDF files
    ├── [paper_id_2]/
    │   └── paper_info.json  # Semantic Scholar info
    └── ...
```

## 🚀 Quick Start

### 1. Set Up the Local Database

First, populate the database with AI/ML papers:

```bash
# Run the setup script to collect 50+ papers
python scripts/setup_local_database.py
```

This will:
- Create the SQLite database
- Download papers from arXiv and Semantic Scholar  
- Store PDFs in individual folders
- Set up the search index

### 2. Launch the Research Assistant

```bash
# Launch the GUI with local database
python launch_gui.py
```

The system now uses the local database instead of external APIs.

## 📊 Database Structure

### Papers Table Schema

```sql
CREATE TABLE papers (
    id TEXT PRIMARY KEY,                    -- Unique paper ID
    title TEXT NOT NULL,                    -- Paper title
    authors TEXT NOT NULL,                  -- JSON array of authors
    abstract TEXT NOT NULL,                 -- Paper abstract
    source TEXT NOT NULL,                   -- 'arxiv' or 'semantic_scholar'
    url TEXT NOT NULL,                      -- Original paper URL
    pdf_path TEXT NOT NULL,                 -- Local PDF file path
    folder_path TEXT NOT NULL,              -- Paper's folder path
    published_date TEXT,                    -- Publication date
    doi TEXT,                               -- DOI if available
    categories TEXT,                        -- JSON array of categories
    keywords TEXT,                          -- JSON array of keywords
    citation_count INTEGER,                 -- Citation count
    venue TEXT,                             -- Publication venue
    arxiv_id TEXT,                          -- arXiv ID if applicable
    semantic_scholar_id TEXT,               -- Semantic Scholar ID if applicable
    created_at TEXT NOT NULL,               -- Database creation timestamp
    updated_at TEXT NOT NULL                -- Last update timestamp
);
```

### File Organization

```
data/papers/
├── papers.db                           # Main database file
├── a1b2c3d4e5f6/                      # Paper folder (ID: a1b2c3d4e5f6)
│   ├── 2301.12345.pdf                 # arXiv PDF
│   └── metadata.json                  # Additional metadata
├── f6e5d4c3b2a1/                      # Another paper folder
│   ├── paper_info.json               # Semantic Scholar info
│   └── supplementary/                 # Optional supplementary files
└── ...
```

## 🔍 Local Search System

### Search Interface

The local search system maintains the same interface as the original API system:

```python
from database.local_search import LocalSearchEngine

# Initialize search engine
search_engine = LocalSearchEngine()

# Search papers
results = await search_engine.search_literature(
    query="machine learning optimization",
    sources=['arxiv'],  # Optional: filter by source
    max_results_per_source=10,
    domain='ai_ml'
)
```

### Domain-Aware Routing

The system intelligently routes queries based on domain detection:

- **AI/ML queries** → Prefer arXiv (recent research)
- **Healthcare queries** → Prefer Semantic Scholar (academic papers)
- **Physics/Math queries** → Prefer arXiv (strong in these fields)

## 🤖 Enhanced RAG System

### Local RAG Pipeline

```python
from rag.local_rag_pipeline import LocalRAGPipeline

# Initialize local RAG
rag = LocalRAGPipeline("data/papers/papers.db")

# Process query with enhanced content generation
response = await rag.process_query(
    query="deep learning for computer vision",
    sources=['arxiv', 'semantic_scholar'],
    generate_hypotheses=True
)

print(response.summary)        # Enhanced summary
print(response.hypotheses)     # Generated hypotheses
```

### Enhanced Content Generation

The system provides:

1. **Intelligent Paper Retrieval** - Relevance scoring and source diversity
2. **LLM-Powered Summarization** - Comprehensive research summaries
3. **Hypothesis Generation** - Novel research directions
4. **Cross-Paper Synthesis** - Insights across multiple papers

## 📈 Collection Statistics

After setup, you can check database statistics:

```python
from database.paper_db import PaperDatabase

db = PaperDatabase()
stats = db.get_stats()

print(f"Total papers: {stats['total_papers']}")
print(f"Papers with PDFs: {stats['papers_with_pdf']}")
print(f"Source distribution: {stats['source_distribution']}")
```

## 🔧 Configuration

### Customizing Paper Collection

Edit `src/database/paper_collector.py` to modify:

- **Search queries** for different AI/ML domains
- **Paper count targets** (default: 50+ papers)
- **Collection strategies** (balanced vs focused)
- **Source preferences** (arXiv vs Semantic Scholar ratio)

### Adjusting Search Relevance

Modify `src/database/local_search.py` to adjust:

- **Relevance scoring algorithms**
- **Domain keyword mappings**
- **Source routing logic**
- **Result ranking strategies**

## 🎯 Benefits of Local Database System

### 1. **Complete Data Control**
- All papers stored locally
- No dependency on external API availability
- Custom paper curation possible

### 2. **Enhanced Privacy**
- No external API calls during operation
- User queries stay local
- Research data remains private

### 3. **Improved Performance**
- Fast local database queries
- No API rate limiting
- Offline capability

### 4. **Advanced Features**
- Custom relevance scoring
- Cross-paper analysis
- Enhanced content generation
- Domain-specific optimizations

### 5. **Research Quality**
- Curated high-quality AI/ML papers
- Balanced source representation
- Recent and relevant research

## 🛠️ Troubleshooting

### Database Issues

```bash
# Check database status
python -c "
from src.database.paper_db import PaperDatabase
db = PaperDatabase()
print(db.get_stats())
"

# Rebuild database if needed
rm data/papers/papers.db
python scripts/setup_local_database.py
```

### Missing PDFs

The system handles cases where PDFs aren't available:
- **arXiv papers**: Direct PDF download
- **Semantic Scholar papers**: Metadata with paper info JSON

### Search Performance

- Database includes optimized indexes
- Relevance scoring is cached
- Results are ranked efficiently

## 🔄 Migration from External APIs

The local system maintains compatibility with the original interface:

- **Same search methods** - `search_literature()` 
- **Same result format** - `SearchResult` objects
- **Same analysis pipeline** - RAG processing
- **Same RL optimization** - Query optimization

Users experience the same functionality with enhanced local performance.

## 📚 Future Enhancements

### Planned Features

1. **Vector Embeddings** - Semantic similarity search
2. **PDF Content Extraction** - Full-text search capabilities  
3. **Citation Network Analysis** - Paper relationship mapping
4. **Custom Collections** - User-defined paper sets
5. **Incremental Updates** - Add new papers over time

### Scalability

- SQLite handles 10,000+ papers efficiently
- Individual folders support large PDF collections
- Modular design allows easy scaling

---

## 📞 Support

For issues with the local database system:

1. Check the **Collection Logs** - `paper_collection.log`
2. Verify **Database Integrity** - Run stats check
3. Review **File Permissions** - Ensure write access to data/papers/
4. Test **Search Functionality** - Use the test queries in setup script

The local database system provides a robust, private, and enhanced research experience with complete control over your AI/ML paper collection. 