from enum import Enum 

class ScrapeStatus(str, Enum):
    pending = 'pending'
    success = 'success'
    failed = 'failed'
    nores = 'no result'

class ScrapeProvider(str, Enum):
    apify = 'apify'