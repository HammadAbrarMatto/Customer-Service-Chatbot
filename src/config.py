"""
config.py
---------
One place to keep all settings used across the whole project.

How to use:
- Make a file called ".env" in the project root (same folder as README.md).
- Put this line inside it:  GOOGLE_API_KEY="your_key_here"
- Get a free key from https://aistudio.google.com/app/apikey
"""

import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. Create a .env file in the project root "
        'and add the line:  GOOGLE_API_KEY="your_key_here"'
    )

CHAT_MODEL = "gemini-2.5-flash"
VISION_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"

# FIX: use absolute paths so it works regardless of working directory
FAQ_DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset", "dataset.csv")
VECTORDB_PATH = os.path.join(os.path.dirname(__file__), "faiss_index")