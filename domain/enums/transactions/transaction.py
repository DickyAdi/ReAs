from enum import Enum

class TransactionStatus(str, Enum):
    pending = 'pending'
    paid = 'paid'
    failed = 'failed'