# ruff: noqa: E402
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI
from contextlib import asynccontextmanager
from slowapi.errors import RateLimitExceeded

from .routes import router
from .context.startup import load_model, get_executor
from .context.limiter import limiter
from .core.handler.exception import rate_limit_handler
from .core.handler import error_handler, unexpected_exception_handler
from .middleware import LoggingMiddleware, CORSMiddleware
from config.settings import settings
from domain.exceptions import BaseError


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    app.state.executor = get_executor()
    yield
    if hasattr(app.state, "model"):
        del app.state.model


app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter

app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.add_exception_handler(RateLimitExceeded, handler=rate_limit_handler)
app.add_exception_handler(BaseError, handler=error_handler)
app.add_exception_handler(Exception, handler=unexpected_exception_handler)
