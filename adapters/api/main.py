# ruff: noqa: E402
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI
from contextlib import asynccontextmanager

from .routes import router
from .context import start_app, stop_app

from .core.handler import error_handler, unexpected_exception_handler
from .middleware import LoggingMiddleware, CORSMiddleware

from config.settings import settings
from domain.exceptions import BaseError


@asynccontextmanager
async def lifespan(app: FastAPI):
    await start_app()

    yield

    await stop_app()


app = FastAPI(lifespan=lifespan)

app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.add_exception_handler(BaseError, handler=error_handler)
app.add_exception_handler(Exception, handler=unexpected_exception_handler)
