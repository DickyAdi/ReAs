from enum import Enum

class Provider(str, Enum):
    google = 'google'
    local = 'local'