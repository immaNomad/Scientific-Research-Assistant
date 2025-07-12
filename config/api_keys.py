"""
Local-Only Configuration - No External APIs
Complete privacy and control using only local models
"""

# EXTERNAL API KEYS DISABLED - USING LOCAL MODELS ONLY
GOOGLE_API_KEY = None  # Disabled - using local model instead
OPENAI_API_KEY = None  # Disabled - using local model instead
ANTHROPIC_API_KEY = None  # Disabled - using local model instead
HUGGING_FACE_API_KEY = None  # Disabled - using local model instead
COHERE_API_KEY = None  # Disabled - using local model instead

# API keys that can stay (for data sources only, not AI models)
PUBMED_API_KEY = None  # Free tier works fine
SEMANTIC_SCHOLAR_API_KEY = None  # Free tier works fine

# FORCE LOCAL MODELS ONLY
USE_LOCAL_MODELS_ONLY = True  # Complete local mode
USE_EXTERNAL_APIS = False  # Disable all external AI APIs

# LOCAL MODEL CONFIGURATION
DEFAULT_CHAT_MODEL = "local"  # Use local AI model
DEFAULT_EMBEDDING_MODEL = "local"  # Use local embeddings
DEFAULT_SEARCH_MODE = "local"  # Use local database only

# LOCAL AI MODEL PATHS
LOCAL_AI_MODEL_PATH = "data/models/local_ai/local_ai_model.pt"
LOCAL_AI_BEST_MODEL_PATH = "data/models/local_ai/best_model.pt" 