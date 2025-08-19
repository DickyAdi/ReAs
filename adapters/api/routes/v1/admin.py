from fastapi import APIRouter, status, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, InterfaceError, OperationalError

from domain.enums.users import Role, AuthProvider
from application.use_cases.subscription import UserRegistrationFlow
from application.user import UserApplication
from infrastructure.db import get_db, User
from loggers.log import get_loggers

from ...schemas import CreateUserRequest, GetUsersResponse
from ...schemas.admin import GetMeResponse
from ...services import min_role, get_user_from_scheme
from ...services.user import get_registration_flow, get_user_application

router = APIRouter(
    prefix="/admin",
    tags=["user/admin"],
    dependencies=[Depends(min_role(min_role=Role.admin))],
)
logger = get_loggers("reas.adapters.api.admin")


@router.post("/create-user")
async def create_user(
    request: CreateUserRequest,
    db: AsyncSession = Depends(get_db),
    admin_creds: User = Depends(min_role(min_role=Role.admin)),
    flow: UserRegistrationFlow = Depends(get_registration_flow),
):
    try:
        new_user = await flow(
            db=db,
            name=request.name,
            email=request.email,
            password=request.password,
            provider=AuthProvider.local,
            idempotency_key=request.idempotency_key,
            is_validated=request.is_validated,
        )
        if new_user:
            logger.info(
                "Admin Name: %s created user with name of %s",
                admin_creds.name,
                new_user.name,
            )
            return JSONResponse(
                status_code=status.HTTP_200_OK, content={"message": "User created."}
            )
    except IntegrityError:
        logger.warning(
            "Admin name: %s tried to create new user but failed", admin_creds.name
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Email or name already exists."
        )
    except OperationalError as e:
        logger.critical(
            "Cannot reach database, check database configuration. %s", str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        )
    except InterfaceError:
        logger.error("Database adapter error cannot finish task.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        )
    except Exception as e:
        logger.warning(
            "Admin name: %s tried to create new user but failed", admin_creds.name
        )
        logger.error("Something went wrong %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong.",
        )


@router.get("/users")
async def list_users(
    offset: int = 0,
    limit: int = 10,
    role: Role = Role.user,
    db: AsyncSession = Depends(get_db),
    user_app: UserApplication = Depends(get_user_application),
    curr_admin: User = Depends(min_role(Role.admin)),
) -> GetUsersResponse:
    if not isinstance(offset, int) or not isinstance(limit, int):
        logger.warning(
            "Wrong query parameter detected, expected Integer and Integer. Got %s and %s",
            type(offset),
            type(limit),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request."
        )
    if role != Role.user and curr_admin.role != Role.superadmin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role."
        )
    try:
        resp = await user_app.get_users(
            db=db, offset=offset, limit=limit, role=role, with_sub=True
        )
    except Exception as e:
        logger.error("Something went wrong: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong",
        )
    return resp


@router.get("/me")
async def get_admin(
    user: User = Depends(get_user_from_scheme(Role.admin)),
) -> GetMeResponse:
    return user
