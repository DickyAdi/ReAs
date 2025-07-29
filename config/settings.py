import os

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field, PostgresDsn
from pydantic_core import MultiHostUrl
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
    jwt_secret_key:str = Field('defaultsecretkeyeventhisisnotpossibletohappenedtho', alias="JWT_SECRET_KEY")
    jwt_algorithm:str = Field('HS256', alias='JWT_ALGORITHM')

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

class DevSettings(Settings):
    postgres_user:str = Field('myuser', alias='DEV_POSTGRES_USER')
    postgres_password:str = Field('mypassword', alias='DEV_POSTGRES_PASSWORD')
    postgres_db:str = Field('mydb', alias='DEV_POSTGRES_DB')
    postgres_host:str = Field('mydb', alias='DEV_POSTGRES_HOST')
    postgres_port:int = Field(5432, alias='DEV_POSTGRES_PORT')

    @computed_field
    @property
    def sqlalchemy_url(self) -> PostgresDsn:
        return MultiHostUrl.build(
            scheme="postgresql+psycopg",
            username=self.postgres_user,
            password=self.postgres_password,
            host=self.postgres_host,
            port=self.postgres_port,
            path=self.postgres_db,
        )

class ProdSettings(Settings):
    postgres_user:str = Field('myuser', alias='PROD_POSTGRES_USER')
    postgres_password:str = Field('mypassword', alias='PROD_POSTGRES_PASSWORD')
    postgres_db:str = Field('mydb', alias='PROD_POSTGRES_DB')
    postgres_host:str = Field('mydb', alias='PROD_POSTGRES_HOST')
    postgres_port:int = Field(5432, alias='PROD_POSTGRES_PORT')

    @computed_field
    @property
    def sqlalchemy_url(self) -> PostgresDsn:
        return MultiHostUrl.build(
            scheme="postgresql+psycopg",
            username=self.postgres_user,
            password=self.postgres_password,
            host=self.postgres_host,
            port=self.postgres_port,
            path=self.postgres_db,
        )

env = os.getenv("ENV", "DEV")

if env == "PROD":
    settings = ProdSettings()
else:
    settings = DevSettings()