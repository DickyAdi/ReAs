from fastapi import APIRouter, status, Request, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated
from ua_parser import parse_device, parse_os, parse_user_agent

from application.use_cases.authentication import (
    ForgetPasswordFlow,
    ResetPasswordFlow,
    UserLoginFlow,
)

from infrastructure.db import get_db

from ...schemas import Token, ForgetPasswordRequest, ResetPasswordRequest
from ...services import (
    get_login_flow,
    get_forget_password_flow,
    get_reset_password_flow,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token")
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: AsyncSession = Depends(get_db),
    flow: UserLoginFlow = Depends(get_login_flow),
) -> Token:
    token = await flow(db=db, email=form_data.username, password=form_data.password)
    return Token(**token.__dict__)


@router.post("/admin/token")
async def login_admin(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: AsyncSession = Depends(get_db),
    flow: UserLoginFlow = Depends(get_login_flow),
):
    token = await flow(
        db=db, email=form_data.username, password=form_data.password, role=None
    )
    return Token(**token.__dict__)


# @router.post('/admin/token')
# async def login_superadmin(form_data:Annotated[OAuth2PasswordRequestForm, Depends()], db:AsyncSession=Depends(get_db), flow:UserLoginFlow=Depends(get_login_flow)):
#     try:
#         Token = await flow(db=db, email=form_data.username, password=form_data.password, role=Role.superadmin)
#     except UserNotFoundException:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect email or password",
#             headers={'WWW-Authenticate' : "Bearer"}
#         )

# @router.post('/superadmin/token')
# async def login_superadmin(form_data:Annotated[OAuth2PasswordRequestForm, Depends()], db:AsyncSession=Depends(get_db), flow:SuperadminLoginFlow=Depends(get_superadmin_login_flow)):
#     try:
#         token = await flow(db=db, email=form_data.username, password=form_data.password)
#     except UserNotFoundException:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect email or password",
#             headers={'Authenticate' : "Bearer"}
#         )
#     return Token(**token.__dict__)


@router.post("/forget-password")
async def forget_password(
    request: Request,
    req: ForgetPasswordRequest,
    db: AsyncSession = Depends(get_db),
    flow: ForgetPasswordFlow = Depends(get_forget_password_flow),
):
    user_agent = request.headers.get("user-agent", "")
    if not req.email or not user_agent:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Detected no user agent or email is empty.",
        )
    requester_device = (
        f"{parse_device(ua=user_agent).brand}; {parse_device(ua=user_agent).family}"
    )
    requester_os = parse_os(ua=user_agent).family
    requester_ua = parse_user_agent(ua=user_agent).family
    ua = f"{requester_device}/{requester_os}/{requester_ua}"
    sent_email = await flow(db=db, user_agent=ua, email=req.email)
    if sent_email:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Forget password request accepted, check user mailbox."
            },
        )


@router.post("/reset-password")
async def reset_password(
    request: ResetPasswordRequest,
    token: str = Query(...),
    db: AsyncSession = Depends(get_db),
    flow: ResetPasswordFlow = Depends(get_reset_password_flow),
):
    changed_password = request.password
    resetted = await flow(db=db, token=token, password=changed_password)
    if resetted:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "Password has been reset and changed."},
        )
