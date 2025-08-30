from fastapi import APIRouter, status, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession


from application.user import UserApplication
from application.auth import AuthApplication
from infrastructure.db import get_db, User
from domain.enums.users import AuthProvider, Role

from application.use_cases.subscription import UserRegistrationFlow
from application.use_cases.authentication import SendVerifyEmailFlow, VerifyEmailFlow
from application.subscriptions import SubscriptionsApplication

from ...schemas import (
    RegisterRequest,
    GetMeResponse,
    EditMeRequest,
    ChangeEmailRequest,
    ChangePasswordRequest,
)
from ...services import (
    get_registration_flow,
    get_user_application,
    get_subscription_application,
    get_send_verify_email_flow,
    get_verify_email_flow,
    get_auth_application,
    get_user_from_scheme,
)

router = APIRouter(tags=["user"])


@router.post("/register")
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db),
    flow: UserRegistrationFlow = Depends(get_registration_flow),
):
    new_acc = await flow(
        db=db,
        name=request.name,
        email=request.email,
        password=request.password,
        provider=AuthProvider.local,
        idempotency_key=request.idempotency_key,
        is_validated=False,
    )
    if new_acc:
        return JSONResponse(
            status_code=status.HTTP_200_OK, content={"message": "success."}
        )


@router.get("/me")
async def get_me(
    current_user: User = Depends(get_user_from_scheme(schema=Role.user)),
    db: AsyncSession = Depends(get_db),
    subs_app: SubscriptionsApplication = Depends(get_subscription_application),
) -> GetMeResponse:
    user_sub = await subs_app.get_user_subscriptions(
        db=db, user_id=current_user.id, is_latestActive=True
    )
    return GetMeResponse(subscription_id=user_sub.id, **current_user.__dict__)


@router.patch("/edit")
async def edit_me(
    request: EditMeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_user_from_scheme),
    flow: UserApplication = Depends(get_user_application),
):
    _updated_user = await flow.edit_user_info(
        db=db, user=current_user, changed_data=request
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK, content={"message": "User edited."}
    )


@router.post("/get-verify-token")
async def verify_email(
    current_user: User = Depends(get_user_from_scheme),
    flow: SendVerifyEmailFlow = Depends(get_send_verify_email_flow),
):
    _sent = await flow(user=current_user)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "Email sent, check user mailbox."},
    )


@router.post("/verify-email-token", dependencies=[Depends(get_user_from_scheme)])
async def decode_email_token(
    token: str = Query(...),
    db: AsyncSession = Depends(get_db),
    flow: VerifyEmailFlow = Depends(get_verify_email_flow),
):
    _updated_user = await flow(db=db, token=token)
    return JSONResponse(
        status_code=status.HTTP_200_OK, content={"message": "User validated."}
    )


@router.patch("/change-email")
async def change_email(
    request: ChangeEmailRequest,
    current_user: User = Depends(get_user_from_scheme),
    db: AsyncSession = Depends(get_db),
    flow: UserApplication = Depends(get_user_application),
):
    changed_email = request.email
    _edited = await flow.change_user_email(
        db=db, changed_email_value=changed_email, user=current_user
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK, content={"message": "Email changed."}
    )


@router.patch("/edit-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(get_user_from_scheme),
    db: AsyncSession = Depends(get_db),
    user_app: UserApplication = Depends(get_user_application),
    auth_app: AuthApplication = Depends(get_auth_application),
):
    hashed_changed_password = auth_app.hash_password(value=request.password)
    _edited = await user_app.change_user_password(
        db=db, user=current_user, hashed_changed_password_value=hashed_changed_password
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK, content={"message": "Password changed."}
    )
