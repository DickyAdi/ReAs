from fastapi import APIRouter, status, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from domain.enums.users import Role, AuthProvider
from application.use_cases.subscription import UserRegistrationFlow
from application.user import UserApplication
from infrastructure.db import get_db, User
from loggers.log import get_loggers

from domain.exceptions import InsufficientRoleError, PaginationTypeError

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
        logger.info(
            "Admin name: %s tried to create new user but failed", admin_creds.name
        )
        raise


@router.get("/users")
async def list_users(
    offset: int = 0,
    limit: int = 10,
    role: Role = Role.user,
    db: AsyncSession = Depends(get_db),
    user_app: UserApplication = Depends(get_user_application),
    curr_admin: User = Depends(min_role(Role.admin)),
) -> GetUsersResponse:
    if offset < 0:
        raise PaginationTypeError(
            message="`offset` must not be less than 0", offset=offset
        )
    if limit < 0 or limit > 100:
        raise PaginationTypeError(message="`limit` must between 1 and 100", limit=limit)
    if (
        role != Role.user and curr_admin.role != Role.superadmin
    ):  # * filter by role, only superadmin can apply filter other than user.
        raise InsufficientRoleError(
            role=curr_admin.role.value, minimum_role=Role.superadmin.value
        )
    resp = await user_app.get_users(
        db=db, offset=offset, limit=limit, role=role, with_sub=True
    )
    return resp


@router.get("/me")
async def get_admin(
    user: User = Depends(get_user_from_scheme(Role.admin)),
) -> GetMeResponse:
    return user
