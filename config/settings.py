# ruff: noqa:E402
import os
from dotenv import load_dotenv

load_dotenv()

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field, PostgresDsn, RedisDsn
from pydantic_core import MultiHostUrl
from typing import Optional


class Settings(BaseSettings):
    max_size_mb: int = Field(5, alias="MAX_CLIENT_UPLOAD_SIZE")
    env: str = Field("DEV", alias="ENV")
    allowed_origins_raw: Optional[str] = Field(None, alias="ALLOWED_ORIGINS_RAW")
    log_dir: str = Field("logs", alias="LOG_DIR")
    log_file: str = Field("reas_app.log", alias="LOG_FILE")
    log_size: int = Field(5, alias="LOG_SIZE")
    predict_chunk_size: int = Field(64, alias="PREDICT_CHUNK_SIZE")
    concurrent_executor: int = Field(2, alias="CONCURRENT_EXECUTOR")
    jwt_secret_key: str = Field(
        "defaultsecretkeyeventhisisnotpossibletohappenedtho", alias="JWT_SECRET_KEY"
    )
    jwt_algorithm: str = Field("HS256", alias="JWT_ALGORITHM")
    n_topic_store: int = Field(50, alias="N_TOPIC_STORE")
    redis_n_connection_pool: int = Field(10, alias="REDIS_N_CONNECTION_POOL")
    scraping_auth_key: str = Field("noneexistence", alias="SCRAPING_AUTH_KEY")

    @computed_field
    @property
    def db_log(self) -> bool:
        if self.env == "DEV":
            return True
        else:
            return False

    @property
    def concurrent_worker(self):
        if self.env == "DEV":
            return 4
        elif self.env == "PROD":
            return self.concurrent_executor

    @property
    def max_size_bytes(self):
        return self.max_size_mb * 1024 * 1024

    @property
    def allowed_origins(self):
        if self.env == "DEV":
            return ["*"]
        if self.allowed_origins_raw:
            return [origin.strip() for origin in self.allowed_origins_raw.split(",")]
        return []

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", populate_by_name=True
    )


class DevSettings(Settings):
    redis_password: str = Field("default", alias="DEV_REDIS_PASSWORD")
    redis_port: int = Field(6379, alias="DEV_REDIS_PORT")
    redis_host: str = Field("localhost", alias="DEV_REDIS_HOST")
    postgres_user: str = Field("myuser", alias="DEV_POSTGRES_USER")
    postgres_password: str = Field("mypassword", alias="DEV_POSTGRES_PASSWORD")
    postgres_db: str = Field("mydb", alias="DEV_POSTGRES_DB")
    postgres_host: str = Field("mydb", alias="DEV_POSTGRES_HOST")
    postgres_port: int = Field(5432, alias="DEV_POSTGRES_PORT")
    smtp_host: str = Field("default", alias="DEV_SMTP_HOST")
    smtp_username: str = Field("default", alias="DEV_SMTP_USERNAME")
    smtp_password: str = Field("default", alias="DEV_SMTP_PASSWORD")
    admin_email: str = Field("reas.admin@reas.com", alias="DEV_ADMIN_USERNAME")
    admin_password: str = Field("adminpassword123", alias="DEV_ADMIN_PASSWORD")
    superadmin_email: str = Field(
        "reas.superadmin@reas.com", alias="DEV_SUPERADMIN_USERNAME"
    )
    superadmin_password: str = Field(
        "superadminpassword123", alias="DEV_SUPERADMIN_PASSWORD"
    )
    domain: str = Field("default", alias="DEV_DOMAIN")

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

    @computed_field
    @property
    def redis_url(self) -> RedisDsn:
        return MultiHostUrl.build(
            scheme="redis",
            password=self.redis_password,
            host=self.redis_host,
            port=self.redis_port,
            path="0",
        ).unicode_string()

    @computed_field
    @property
    def celery_broker_url(self) -> RedisDsn:
        return MultiHostUrl.build(
            scheme="redis",
            password=self.redis_password,
            host=self.redis_host,
            port=self.redis_port,
            path="1",
        ).unicode_string()

    @computed_field
    @property
    def celery_backend_url(self) -> RedisDsn:
        return MultiHostUrl.build(
            scheme="redis",
            password=self.redis_password,
            host=self.redis_host,
            port=self.redis_port,
            path="2",
        ).unicode_string()


class ProdSettings(Settings):
    redis_password: str = Field("default", alias="PROD_REDIS_PASSWORD")
    redis_port: int = Field(6379, alias="PROD_REDIS_PORT")
    redis_host: str = Field("localhost", alias="PROD_REDIS_HOST")
    postgres_user: str = Field("myuser", alias="PROD_POSTGRES_USER")
    postgres_password: str = Field("mypassword", alias="PROD_POSTGRES_PASSWORD")
    postgres_db: str = Field("mydb", alias="PROD_POSTGRES_DB")
    postgres_host: str = Field("mydb", alias="PROD_POSTGRES_HOST")
    postgres_port: int = Field(5432, alias="PROD_POSTGRES_PORT")
    smtp_host: str = Field("default", alias="PROD_SMTP_HOST")
    smtp_username: str = Field("default", alias="PROD_SMTP_USERNAME")
    smtp_password: str = Field("default", alias="PROD_SMTP_PASSWORD")
    admin_email: str = Field("reas.admin@reas.com", alias="PROD_ADMIN_USERNAME")
    admin_password: str = Field("adminpassword123", alias="PROD_ADMIN_PASSWORD")
    superadmin_email: str = Field(
        "reas.superadmin@reas.com", alias="PROD_SUPERADMIN_USERNAME"
    )
    superadmin_password: str = Field(
        "superadminpassword123", alias="PROD_SUPERADMIN_PASSWORD"
    )
    domain: str = Field("default", alias="PROD_DOMAIN")

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

    @computed_field
    @property
    def redis_url(self) -> RedisDsn:
        return MultiHostUrl.build(
            scheme="redis",
            password=self.redis_password,
            host=self.redis_host,
            port=self.redis_port,
            path="0",
        ).unicode_string()

    @computed_field
    @property
    def celery_broker_url(self) -> RedisDsn:
        return MultiHostUrl.build(
            scheme="redis",
            password=self.redis_password,
            host=self.redis_host,
            port=self.redis_port,
            path="1",
        ).unicode_string()

    @computed_field
    @property
    def celery_backend_url(self) -> RedisDsn:
        return MultiHostUrl.build(
            scheme="redis",
            password=self.redis_password,
            host=self.redis_host,
            port=self.redis_port,
            path="2",
        ).unicode_string()


env = os.getenv("ENV", "DEV")

if env == "PROD":
    settings = ProdSettings()
else:
    settings = DevSettings()
