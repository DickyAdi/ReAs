from enum import Enum 

class UserStatus(str, Enum):
    active = 'active'
    deactivated = 'deactivated'
    deleted = 'deleted'