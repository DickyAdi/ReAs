from abc import ABC, abstractmethod
from domain.entities.user import UserEntities
from typing import Optional

class UserInterface(ABC):
    @abstractmethod
    async def get_user_by_email(self, db, email:str):
        pass
    @abstractmethod
    async def get_users(self, db, offset:int, limit:int):
        pass
    @abstractmethod
    async def create_user(self, db, user:UserEntities):
        pass
    @abstractmethod
    async def delete_user(self, db, email:Optional[str]=None, user_id:Optional[int]=None):
        pass