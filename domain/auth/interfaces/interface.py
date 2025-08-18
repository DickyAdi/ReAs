from datetime import timedelta
from abc import ABC, abstractmethod

class AuthInterface(ABC):
    @abstractmethod
    def create_access_token(self, data:dict, expires_time:timedelta):
        pass
    @abstractmethod
    async def decode_user_token(self, token, db):
        pass
    @abstractmethod
    def verify_password(self, user_password, db_password):
        pass
    @abstractmethod
    def decode_token(self, token):
        pass
    @abstractmethod
    def hash(self, value):
        pass