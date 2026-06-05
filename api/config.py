from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    blog_base_url: str = "https://www.company.com"
    chroma_db_path: str = "./chroma_db"
    collection_name: str = "blog_articles"
    llm_model: str = "gpt-4o-mini"
    classifier_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
