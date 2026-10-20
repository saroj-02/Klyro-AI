
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Klyro-AI"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Select "local" or "cloud"
    llm_provider: str = "local"

    # Local Ollama configuration
    ollama_base_url: str = "http://localhost:11434"

    # Ollama Cloud configuration
    ollama_api_key: str = ""
    ollama_cloud_base_url: str = "https://ollama.com/v1"
    ollama_model: str = "llama3.2:3b"

    # Retrieval models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    data_dir: Path = Path("./data")
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_top_k: int = 8
    rerank_top_k: int = 4
    min_relevance_score: float = 0.15
    max_upload_mb: int = 100

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    return settings
