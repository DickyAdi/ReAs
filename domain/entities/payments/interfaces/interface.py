from abc import ABC, abstractmethod
from typing import Optional

from domain.entities.payments import PaymentEntity

class PaymentInterface(ABC):
    @abstractmethod
    async def create_payment(self, db, payment:PaymentEntity, commit:bool=True):
        pass

    @abstractmethod
    async def get_payment(self, db):
        pass