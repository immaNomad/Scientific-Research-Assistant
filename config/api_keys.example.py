# Google API Key (for Gemini models)
GOOGLE_API_KEY = "AIzaSyCoPpl_gokRRupcMFd5B1SpAZinQxWmEpQ"

# PubMed API Key (optional, for higher rate limits)
PUBMED_API_KEY = None  # Free tier works fine

# Semantic Scholar API Key (optional, for higher rate limits)
SEMANTIC_SCHOLAR_API_KEY = None  # Free tier works fine

# Set to True to use free/local models only (no API calls)
USE_LOCAL_MODELS_ONLY = False

# Default model preferences
DEFAULT_CHAT_MODEL = "gemini-pro"  # Using Google's Gemini
DEFAULT_EMBEDDING_MODEL = "local"  # Use local embeddings since no OpenAI key 