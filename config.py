import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

LLM_MODEL = "gemini-3.1-flash-lite"

QUERY_REWRITE_ENABLED = False

QUERY_REWRITE_MODEL = LLM_MODEL

CHUNK_SIZE = 200

CHUNK_OVERLAP = 30

TOP_K = 3
