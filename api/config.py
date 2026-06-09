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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,  # blank .env entry won't clobber a default
        case_sensitive=False,  # OPENAI_API_KEY -> openai_api_key (explicit)
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
