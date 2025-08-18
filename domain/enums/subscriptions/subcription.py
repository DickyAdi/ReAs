from enum import Enum

class SubscriptionStatus(str, Enum):
    active = 'active'
    pending = 'pending'
    expired = 'expired'
    freeze = 'freeze'
    trial = 'trial'
    canceled = 'canceled'

class SubscriptionPlan(str, Enum):
    monthly = 'monthly'
    annually = 'annually'
    free = 'free'