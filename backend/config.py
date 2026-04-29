# Configuration module
# Loads environment variables and application settings
"""
MumzSense v1 — Configuration Module
"""
import os
from functools import lru_cache
from pathlib import Path

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


def _find_env_file() -> str:
    """Search common locations for .env file."""
    candidates = [
        Path(__file__).parent / ".env",
        Path(__file__).parent.parent / "env" / ".env",
        Path(__file__).parent.parent / ".env",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return ".env"


class Settings(BaseSettings):
    # LLM — supports "groq_api" | "groq" | "openrouter" | "llama_api"
    llm_provider: str = "groq_api"
    llama_api_key: str = ""          # used when llm_provider == "llama_api"
    groq_api_key: str = ""           # used when llm_provider == "groq_api" or "groq"
    llama_model: str = "meta-llama/Llama-3.1-70B-Instruct"
    llm_max_tokens: int = 600
    llm_temperature: float = 0.4
    llm_top_p: float = 0.85

    # Supabase
    supabase_url: str = ""
    supabase_key: str = ""
    database_url: str = ""

    # Redis
    redis_url: str = ""

    # FastAPI
    fastapi_env: str = "development"
    debug: bool = True
    admin_token: str = "changeme"

    # Embedding
    en_embedding_model: str = "BAAI/bge-large-en-v1.5"
    ar_embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    en_embedding_dim: int = 1024
    ar_embedding_dim: int = 768

    # RAG thresholds
    similarity_high: float = 0.75
    similarity_medium: float = 0.60
    similarity_low: float = 0.45
    rag_top_k: int = 5

    # Classifier
    confidence_threshold: float = 0.45  # lowered from 0.60 — BiLSTM not yet trained
    use_real_embeddings: bool = True     # MUST be True for semantic search

    # App meta
    version: str = "1.0.0"
    corpus_size: int = 560

    class Config:
        env_file = _find_env_file()
        case_sensitive = False
        extra = "ignore"

    def effective_llm_key(self) -> str:
        """Return the correct API key based on llm_provider."""
        normalized = self.llm_provider.lower().replace("_api", "")
        if normalized == "groq":
            return self.groq_api_key or self.llama_api_key
        return self.llama_api_key or self.groq_api_key

    def effective_llm_provider(self) -> str:
        """Normalize provider string to one the agent understands."""
        v = self.llm_provider.lower()
        if v in ("groq", "groq_api"):
            return "groq"
        if v == "openrouter":
            return "openrouter"
        return "llama_api"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()