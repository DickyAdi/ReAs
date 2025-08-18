from fastapi import APIRouter, status, Depends, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, InterfaceError, OperationalError
from typing import Annotated
from decimal import Decimal

from domain.enums.users import Role, AuthProvider
from domain.exceptions import PlanDeactivateError, PlanRefError
from infrastructure.db import get_db, User
from infrastructure.db.models import SubscriptionPlans, Tiers
from loggers.log import get_loggers

from application.subscriptions import SubscriptionPlanApplication
from application.use_cases.subscription import UserRegistrationFlow, CreateNewPlanFlow

# from domain.enums.tiers_pricing import Currency
# from domain.enums.subscriptions import SubscriptionPlan
# from domain.enums.tiers import Tier

from ...services import min_role, get_user_from_any_schema, get_user_from_scheme, get_registration_flow, get_subscription_plan_application, get_create_plan_flow
from ...schemas.admin import CreateAdminRequest, CreatePlanRequestForm

router = APIRouter(prefix='/superadmin', tags=['user/superadmin'], dependencies=[Depends(min_role(min_role=Role.superadmin))])
logger = get_loggers('reas.adapters.api.superadmin')

@router.post('/create-admin')
async def create_admin(request:CreateAdminRequest, db:AsyncSession=Depends(get_db), curr_admin:User=Depends(min_role(Role.superadmin)), flow:UserRegistrationFlow=Depends(get_registration_flow)):
    try:
        new_admin = await flow(db=db, name=request.name, email=request.email, password=request.password, provider=AuthProvider.local, idempotency_key=request.idempotency_key, is_validated=request.is_validated, role=Role.admin)
        if new_admin:
            logger.info('Superadmin name: %s created admin with name of %s', curr_admin.name, new_admin.name)
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    'message' : 'Admin created.'
                }
            )
    except IntegrityError:
        logger.warning('Superadmin name: %s trying to create new admin but failed. ', curr_admin.name)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail='Email or name already exists.'
        )
    except OperationalError as e:
        logger.critical('Cannot reach database, check database configuration. %s', str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error."
        )
    except InterfaceError:
        logger.error('Database adapter error cannot finish task.')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error."
        )
    except Exception as e:
        logger.warning('Admin name: %s tried to create new user but failed', curr_admin.name)
        logger.error('Something went wrong %s', str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong."
        )

@router.get('/plans')
async def list_plans(offset:int=0, limit:int=10, isActive:bool=False, db:AsyncSession=Depends(get_db), plan_app:SubscriptionPlanApplication=Depends(get_subscription_plan_application)):
    if not isinstance(offset, int) or not isinstance(limit, int):
        logger.warning("Wrong query parameter detected, expected Integer and Integer. Got %s and %s", type(offset), type(limit))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Invalid request.'
        )
    try:
        resp = await plan_app.get_plans(db=db, offset=offset, limit=limit, with_tier=True, isActive=isActive)
    except Exception as e:
        logger.error('Something went wrong: %s', str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Something went wrong.'
        )
    return resp

@router.post('/create-plan')
async def create_plans(form_data:Annotated[CreatePlanRequestForm, Form()], db:AsyncSession=Depends(get_db), flow:CreateNewPlanFlow=Depends(get_create_plan_flow)):
    try:
        new_plan = await flow(db=db, tier=form_data.tier, cycle=form_data.cycle, currency=form_data.currency, price=form_data.price, duration_days=form_data.duration_days)
        logger.info('Superadmin created new plan with code of %s', new_plan.code)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                'message' : 'new plan appended.'
            }
        )
    except PlanRefError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='New plan must refer to existing plan.'
        )
    except PlanDeactivateError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Something went wrong in deactivation of existing plan.'
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Something went wrong: {str(e)}"
        )