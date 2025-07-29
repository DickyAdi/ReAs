from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from config.settings import settings

class Base(DeclarativeBase):
    pass

engine = create_async_engine(str(settings.sqlalchemy_url), echo=settings.db_log)
SessionLocal = async_sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        await db.close()
