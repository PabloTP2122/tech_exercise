from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: SecretStr
    blog_base_url: str = "https://www.bitovi.com"
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/blog_rag"
    collection_name: str = "chunks"
    llm_model: str = "gpt-4o-mini"
    classifier_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    similarity_threshold: float = 0.35  # relevance floor (0..1, higher=better); tune via .env
    retrieval_top_k: int = 4  # vector hits fetched per question
    keyword_top_k: int = 4  # full-text hits fused via RRF; 0 disables hybrid search
    max_chunks_per_article: int = 2  # context-diversity cap applied after fusion
    cors_allow_origins: str = "http://localhost:3000,http://localhost:4200"
    # Comma-separated list of allowed CORS origins.  Override at deploy time via env:
    #   CORS_ALLOW_ORIGINS=https://your-app.vercel.app,http://localhost:3000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,  # blank .env entry won't clobber a default
        case_sensitive=False,  # OPENAI_API_KEY -> openai_api_key (explicit)
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
