from dataclasses import dataclass
from uuid import UUID
from typing import Optional, TYPE_CHECKING
from datetime import datetime
from decimal import Decimal

if TYPE_CHECKING:
    from domain.entities.transactions import TransactionEntity
    
from domain.enums.payments import PaymentStatus, PaymentMethod

@dataclass
class PaymentEntity:
    method: PaymentMethod
    status: PaymentStatus
    amount_charged: Decimal
    provider_ref:Optional[str] = None
    created_at: Optional[datetime | None] = None #derived from database
    updated_at: Optional[datetime | None] = None #derived from database
    id: Optional[UUID | None] = None #derived from database

    #related entities
    transaction:Optional["TransactionEntity"] = None

    @classmethod
    def create(cls, method:PaymentMethod, amount_charged:Decimal, status:PaymentStatus, transaction:Optional["TransactionEntity"]=None) -> "PaymentEntity":
        return cls(
            method=method,
            status=status,
            amount_charged=amount_charged,
            transaction=transaction,
        )