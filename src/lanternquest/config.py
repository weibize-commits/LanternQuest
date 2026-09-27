from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Local paths and safe defaults for research artifacts."""

    model_config = SettingsConfigDict(
        env_prefix="LANTERNQUEST_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    source_root: Path = Field(default=Path("汴京灯笼张素材资料"))
    artifact_root: Path = Field(default=Path("artifacts"))
    hash_chunk_mb: int = Field(default=8, ge=1, le=128)
    llm_model: str | None = None
    llm_max_output_tokens: int = Field(default=2000, ge=128, le=32000)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_reasoning_effort: str | None = None
    llm_structured_output_mode: str = "json_schema"
    openai_base_url: str | None = None
    openai_api_key: SecretStr | None = None
    bit_api_key: SecretStr | None = None
    deepseek_api_key: SecretStr | None = None
    kimi_api_key: SecretStr | None = None
    glm_api_key: SecretStr | None = None

    def resolved_source_root(self) -> Path:
        return self.source_root.expanduser().resolve()

    def resolved_artifact_root(self) -> Path:
        return self.artifact_root.expanduser().resolve()
