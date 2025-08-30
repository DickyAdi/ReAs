from fastapi import APIRouter

from .healthcheck import router as healthcheck
from .extract import router as extract
from .user import router as user
from .auth import router as auth
from .admin import router as admin
from .superadmin import router as superadmin
from .dataset import router as dataset

router = APIRouter()

router.include_router(healthcheck)
router.include_router(extract)
router.include_router(user)
router.include_router(auth)
router.include_router(admin)
router.include_router(superadmin)
router.include_router(dataset)
