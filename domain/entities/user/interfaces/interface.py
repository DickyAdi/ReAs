from uuid import UUID
from abc import ABC, abstractmethod
from domain.entities.user import UserEntities
from typing import Optional

class UserInterface(ABC):
    @abstractmethod
    async def get_user_by_email(self, db, email:str, role):
        pass
    @abstractmethod
    async def get_user_by_uuid(self, db, uuid:UUID, role):
        pass
    @abstractmethod
    async def get_users(self, db, offset:int, limit:int, role, with_sub:bool):
        pass
    @abstractmethod
    async def create_user(self, db, user:UserEntities, commit:bool):
        pass
    @abstractmethod
    async def delete_user(self, db, email:Optional[str]=None, user_id:Optional[int]=None):
        pass
    @abstractmethod
    async def edit_user(self, db, user):
        pass
    @abstractmethod
    async def get_user_with_sub(self, db, id:Optional[int], email:Optional[str]):
        pass