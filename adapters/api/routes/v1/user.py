from fastapi import APIRouter, status, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, InterfaceError, OperationalError


from application.user import UserApplication
from infrastructure.db.services import UserService
from infrastructure.db import get_db, User
from loggers.log import get_loggers
from infrastructure.auth import AuthService
from domain.provider import Provider

from ...schemas import RegisterRequest
from ...services import get_current_active_user

router = APIRouter(tags=['user'])
logger = get_loggers('reas.adapters.api')


@router.post('/register')
async def register(
    request:RegisterRequest, 
    db:AsyncSession=Depends(get_db)):


    flow = UserApplication(service=UserService(), auth=AuthService())
    try:
        new_acc = await flow.register(name=request.name, email=request.email, password=request.password, db=db, provider=Provider.local)
        if new_acc:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    'message' : 'success.'
                }
            )
    except IntegrityError:
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email or name already exists."
        )
    except OperationalError as e:
        logger.critical('Cannot reach database, check database configuration. %s', str(e))
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error."
        )
    except InterfaceError:
        logger.error('Database adapter error cannot finish task.')
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error."
        )
    except Exception as e:
        logger.error('Something went wrong %s', str(e))
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong."
        )

@router.get('/me')
def get_me(current_user:User = Depends(get_current_active_user)):
    return current_user