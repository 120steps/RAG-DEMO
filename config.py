import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

LLM_MODEL = "gemini-3.1-flash-lite"