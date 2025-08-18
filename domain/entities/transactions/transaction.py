from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, TypedDict
from datetime import datetime
from uuid import UUID
from decimal import Decimal

if TYPE_CHECKING:
    from domain.entities.subscriptions import SubscriptionEntity
    from domain.entities.payments import PaymentEntity

from domain.enums.transactions import TransactionStatus
from domain.enums.tiers_pricing import Currency

class PlanSnapshotDict(TypedDict):
    code:str
    price:Decimal
    currency: str
    created_at:datetime

@dataclass
class TransactionEntity:
    status: TransactionStatus
    amount: Decimal
    plan_snapshot: PlanSnapshotDict
    payment_id:UUID #derived from database
    idempotency_key: Optional[str] #!ONLY SET IT TO OPTIONAL IN DEVELOPEMNT, WHILE REAS ARE READY TO LAUNCH, THEN REMOVE THE Optional[]
    created_at: Optional[datetime] = None #derived from database
    updated_at: Optional[datetime] = None #derived from database
    id:Optional[UUID] = None #derived from database

    #related entity
    payment:Optional["PaymentEntity"] = None
    subscription:Optional["SubscriptionEntity"] = None

    @classmethod
    def create(cls, status:TransactionStatus, amount:Decimal, payment_id:UUID, plan_snapshot:PlanSnapshotDict, idempotency_key:Optional[UUID], payment:Optional['PaymentEntity']=None, subscription:Optional['SubscriptionEntity']=None) -> "TransactionEntity":
        return cls(
            status=status,
            amount=amount,
            payment_id=payment_id,
            plan_snapshot=plan_snapshot,
            idempotency_key=str(idempotency_key),
            payment=payment,
            subscription=subscription,
        )
    def get_transaction_status(self) -> TransactionStatus:
        return self.status