from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional
import bcrypt

from domain.entities.user import UserEntities
from domain.entities.user.interfaces import UserInterface
from ...db.models import User

class UserService(UserInterface):
    async def get_users(self, db:AsyncSession, offset:int=0, limit:int=0):
    # async def get_users(self, offset:int=0, limit:int=0):
        statement = select(User).offset(offset).limit(limit)
        result = await db.execute(statement)
        users = result.scalars().all()

        if not offset and not limit:
            return users

        count_statement = select(func.count()).select_from(User)
        count_result = await db.execute(count_statement)
        count_value = count_result.scalar_one()
        return {
            "total" : count_value,
            'offset' : offset,
            'limit' : limit,
            'data' : users
        }
    async def get_user_by_email(self, db:AsyncSession, email:str):
    # async def get_user_by_email(self, email:str):
        statement = select(User).where(User.email == email)
        result = await db.execute(statement)
        user = result.scalars().first()
        return user
    
    async def create_user(self, db:AsyncSession, user:UserEntities):
    # async def create_user(self, user:UserEntities):
        new_user = User.from_entity(user)
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        return new_user
    
    async def delete_user(self, db:AsyncSession, email:Optional[str]=None, user_id:Optional[str]=None):
    # async def delete_user(self, email:Optional[str]=None, user_id:Optional[str]=None):
        if email and user_id:
            raise KeyError("Cannot insert new user using 2 identifiers provided. Only 1 identifier must be provided.")
        if email:
            statement = select(User).where(User.email == email)
        elif user_id:
            statement = select(User).where(User.id == user_id)
        result = await db.execute(statement)
        user_db = result.scalar_one_or_none()
        if user_db:
            await db.delete(user_db)
            await db.commit()
        return

