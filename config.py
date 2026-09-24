"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_PATH = Path(os.getenv("SOP_DATA_PATH", BASE_DIR / "data" / "sop.md"))
STORAGE_DIR = Path(os.getenv("SOP_STORAGE_DIR", BASE_DIR / "storage"))
CHROMA_DIR = STORAGE_DIR / "chroma"
SOPS_JSON = STORAGE_DIR / "sops.json"          # full SOP text for the popup viewer
COLLECTION = "sop_chunks"

# Branding
APP_TITLE = "Store SOP Assistant"
ORG_NAME = "A.J. Textile Mills Limited"

# Embeddings
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", 16))
EMBED_MAX_SEQ = int(os.getenv("EMBED_MAX_SEQ", 512))

# Structure-aware chunking
MAX_CHARS = 1400          # soft cap for a text chunk (list items are never split below this)
HARD_MAX_CHARS = 3000     # a single point longer than this is split on sentence boundaries
MAX_TABLE_ROWS = 25       # tables longer than this are split by rows, header repeated

# Retrieval
TOP_K = 6
MIN_SCORE = 0.35          # cosine similarity below which a passage is treated as irrelevant

# LLM (Groq)
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = 0.1
