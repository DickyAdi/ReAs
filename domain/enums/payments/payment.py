from enum import Enum

class PaymentStatus(str, Enum):
    success = 'success'
    failed = 'failed'
    invalid = 'invalid'
    pending = 'pending'

class PaymentMethod(str, Enum):
    local = 'local'