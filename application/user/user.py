from typing import Optional
from datetime import timedelta

from domain.entities.user.interfaces import UserInterface
from domain.auth import AuthInterface, Token
from domain.entities.user import UserEntities, Tier, Provider

class UserApplication:
    def __init__(self, service:UserInterface, auth:AuthInterface):
        self.user_service = service
        self.auth_service = auth
    async def register(self, db, name:str, email:str, password:str, provider:Provider, tier:Optional[Tier]=Tier.Base, is_validated:Optional[bool]=False, is_superuser:Optional[bool]=False):
        new_user = UserEntities.create(name=name, email=email, password=password, tier=tier, is_validated=is_validated, is_superuser=is_superuser, provider=provider)
        created_user = await self.user_service.create_user(user=new_user, db=db)
        return created_user
    async def login_for_access_token(self, db, email:str, password:str, expired_timedelta:timedelta=timedelta(minutes=1440)):
        user = await self.user_service.get_user_by_email(db=db, email=email)
        if not user or not self.auth_service.verify_password(password, user.password):
            raise ValueError(f"Incorrect email or password.")
        access_token = self.auth_service.create_access_token(data={'sub' : user.email}, expires_time=expired_timedelta)
        return Token(access_token=access_token, token_type="bearer")