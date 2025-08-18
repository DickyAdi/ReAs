from abc import ABC, abstractmethod
from typing import Optional

from domain.entities.transactions import TransactionEntity

class TransactionInterface(ABC):
    @abstractmethod
    async def create_idempotent_transaction(self, db, transaction:TransactionEntity, commit:bool=True):
        pass

    @abstractmethod
    async def get_transaction(self, db):
        pass

    @abstractmethod
    async def get_transaction_by_idempotency_key(self, db, idempotency_key):
        pass