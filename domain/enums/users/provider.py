from enum import Enum

class AuthProvider(str, Enum):
    google = 'google'
    local = 'local'