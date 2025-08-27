from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from typing import Annotated, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from application.use_cases.authentication import (
    ForgetPasswordFlow,
    ResetPasswordFlow,
    UserAuthenticateFlow,
)
from application.auth import AuthApplication
from infrastructure.db import get_db, User
from infrastructure.auth import AuthService
from infrastructure.mail import MailtrapService
from infrastructure.db.services import UserService
from domain.enums.users import Role, ROLE_LEVEL
from domain.enums.tiers import Tier, TIER_LEVEL
from domain.exceptions import InsufficientTierError, InsufficientRoleError

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/auth/token", scheme_name="oauth2_scheme"
)
admin_oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/auth/admin/token", scheme_name="admin_oauth2_scheme"
)


def get_auth_application():
    """Get auth application dependency injection style.

    Returns:
        AuthApplication: Auth application class object
    """
    return AuthApplication(service=AuthService())


def get_forget_password_flow():
    """Get forget password flow from use_cases dependency injection style.

    Returns:
        ForgetPasswordFlow: Forget password flow use cases class object.
    """
    return ForgetPasswordFlow(
        user_service=UserService(),
        auth_service=AuthService(),
        mail_service=MailtrapService(),
    )


def get_reset_password_flow():
    """Get reset password flow from use_cases dependency injection style.

    Returns:
        ResetPasswordFlow: Reset password flow uses cases class object.
    """
    return ResetPasswordFlow(user_service=UserService(), auth_service=AuthService())


def get_user_auth_flow():
    """Get user authenticate flow from use_cases dependency injection style.

    Returns:
        UserAuthenticateFlow: User authenticate flow use cases class object
    """
    return UserAuthenticateFlow(user_service=UserService(), auth_service=AuthService())


def get_user_from_scheme(schema: Optional[Role] = Role.user, with_sub: bool = False):
    """Get user based on given scheme. Authentication token will be extracted based on the given scheme.

    Args:
        schema (Optional[Role], optional): Scheme to be checked. Defaults to Role.user.
        with_sub (bool, optional): Whether to include user subscriptions or not. Defaults to False.

    Returns:
        User: Authenticated user
    """
    if schema == Role.user:
        scheme = oauth2_scheme
    else:
        scheme = admin_oauth2_scheme

    async def get_current_user(
        token: Annotated[str, Depends(scheme)],
        db: AsyncSession = Depends(get_db),
        flow: UserAuthenticateFlow = Depends(get_user_auth_flow),
    ):
        user = await flow(db=db, token=token, with_sub=with_sub)
        return user

    return get_current_user


async def get_token_from_any_scheme(request: Request):
    """Get any token scheme from request headers by looping through all available schemas.

    Args:
        request (Request): FastAPI Request annotations. Do not pass any value to this parameter.

    Raises:
        HTTPException: 401 HTTP Exception, when token is not found.

    Returns:
        str: JWT token string
    """
    schemas = [oauth2_scheme, admin_oauth2_scheme]
    for schema in schemas:
        token = await schema(request)
        return token


def get_user_from_any_schema(with_sub: bool = False):
    """Get user from any schema by explicitly checking each token schema.

    Args:
        with_sub (bool, optional): Whether to include user subscription or not. Defaults to False.

    Returns:
        Function: A function that specifically have been set to a correct parameter value. Must be called when using this in dependency injection.
    """

    async def get_current_user(
        token: str = Depends(get_token_from_any_scheme),
        db: AsyncSession = Depends(get_db),
        flow: UserAuthenticateFlow = Depends(get_user_auth_flow),
    ):
        user = await flow(db=db, token=token, with_sub=with_sub)
        return user

    return get_current_user


def min_tier(min_tier: Tier):
    """Tier checking function to let user pass a certain protected routes. User with role admin or higher will automatically pass.

    Args:
        min_tier (Tier): Minimum tier to be able to pass.

    Returns:
        Function: A function that specifically have been set to a correct parameter value, where it automatically returns User.
    """

    async def checker(user: User = Depends(get_user_from_any_schema(with_sub=True))):
        if ROLE_LEVEL[user.role] >= ROLE_LEVEL[Role.admin]:
            return user
        user_tier = user.subscriptions[0].plan.tier.name
        if TIER_LEVEL[user_tier] < TIER_LEVEL[min_tier]:
            raise InsufficientTierError(
                tier=user_tier.value, minimum_tier=min_tier.value
            )
        return user

    return checker


def min_role(min_role: Role):
    """Role checking function to let user pass a certain protected routes. User with role admin or higher will automatically pass.

    Args:
        min_role (Role): Minimum role to be able to pass.

    Returns:
        Function: A function that specifically have been set to a correct parameter value, where it automatically returns User.
    """

    async def checker(user: User = Depends(get_user_from_any_schema())):
        if ROLE_LEVEL[user.role] < ROLE_LEVEL[min_role]:
            raise InsufficientRoleError(
                role=user.role.value, minimum_role=min_role.value
            )
        return user

    return checker
