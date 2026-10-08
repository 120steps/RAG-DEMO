from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from config import GEMINI_API_KEY, LLM_MODEL


@lru_cache(maxsize=1)
def get_chat_model() -> ChatGoogleGenerativeAI:
    """Create the LangChain chat model from the existing Gemini config."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Add it to the project .env "
            "before invoking the LangChain RAG chain."
        )

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        api_key=GEMINI_API_KEY,
        temperature=0,
    )
