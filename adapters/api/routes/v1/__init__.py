from fastapi import APIRouter

from .healthcheck import router as healthcheck
from .extract import router as extract
from .user import router as user
from .auth import router as auth

router = APIRouter()

router.include_router(healthcheck)
router.include_router(extract)
router.include_router(user)
router.include_router(auth)

