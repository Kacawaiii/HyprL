from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the HyprL API service."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "HyprL API"
    api_version: str = "2.0"

    database_url: str = "sqlite:///data/hyprl_api.db"
    token_hash_secret: str = "change-me"
    discord_registration_secret: str = "dev-discord-secret"

    default_max_rpm: int = 30
    default_max_rpd: int = 500
    default_daily_credits: int = 200


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()
