from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    database_url: str = Field("sqlite:///./exovision.db", alias="EV_DATABASE_URL")
    allow_origins: str = Field("http://localhost:5173,http://127.0.0.1:5173", alias="EV_ALLOW_ORIGINS")
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()