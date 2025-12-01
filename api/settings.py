from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    hyprl_api_host: str = "0.0.0.0"
    hyprl_api_port: int = 8000
    hyprl_db_url: str = "sqlite:///./hyprl.db"
    
    # Auth
    hyprl_admin_token: str | None = None
    demo_key: str = "DEMO-KEY"
    allow_free_token: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
