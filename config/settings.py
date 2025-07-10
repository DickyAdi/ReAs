import os

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List, Optional


class Settings(BaseSettings):
    max_size_mb: int = Field(5, alias='MAX_CLIENT_UPLOAD_SIZE')
    env:str = Field('DEV', alias='ENV')
    allowed_origins_raw:Optional[str] = Field(None, alias='ALLOWED_ORIGINS_RAW')
    log_dir:str = Field('logs', alias='LOG_DIR')
    log_file:str = Field('reas_app.log', alias='LOG_FILE')
    log_size:int = Field(5, alias='LOG_SIZE')
    predict_chunk_size:int = Field(64, alias='PREDICT_CHUNK_SIZE')
    concurrent_executor:int = Field(2, alias='CONCURRENT_EXECUTOR')
    
    @property
    def concurrent_worker(self):
        if self.env == 'DEV':
            return 4
        elif self.env == 'PROD':
            return self.concurrent_executor

    @property
    def max_size_bytes(self):
        return self.max_size_mb * 1024 * 1024
    
    @property
    def allowed_origins(self):
        if self.env == 'DEV':
            return ['*']
        if self.allowed_origins_raw:
            return [origin.strip() for origin in self.allowed_origins_raw.split(',')]
        return []

    
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', populate_by_name=True)

settings = Settings()