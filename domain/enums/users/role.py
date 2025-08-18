from enum import Enum

class Role(str, Enum):
    user = 'user'
    admin = 'admin'
    superadmin = 'superadmin'

ROLE_LEVEL = {
    'user' : 0,
    'admin' : 1,
    'superadmin' : 2
}