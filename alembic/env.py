from logging.config import fileConfig
import asyncio

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from infrastructure.db.db import Base
from config.settings import settings
# from infrastructure.db.models import user_model  # Make sure models are imported so metadata is populated
import infrastructure.db.models

# Alembic Config object
config = context.config

# Inject DB URL from your settings dynamically
config.set_main_option("sqlalchemy.url", str(settings.sqlalchemy_url))

# Setup logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the metadata for Alembic to use for 'autogenerate'
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (no DB connection)."""
    context.configure(
        url=str(settings.sqlalchemy_url),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode using async SQLAlchemy engine."""
    connectable = async_engine_from_config(
        {
            "sqlalchemy.url": str(settings.sqlalchemy_url),
        },
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        def do_migrations(sync_connection):
            context.configure(
                connection=sync_connection,
                target_metadata=target_metadata,
                compare_type=True,
            )

            with context.begin_transaction():
                context.run_migrations()

        await connection.run_sync(do_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())