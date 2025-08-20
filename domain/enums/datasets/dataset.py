from enum import Enum


class DatasetStatus(str, Enum):
    pending = "pending"
    success = "success"
    failed = "failed"
    nores = "no result"


class DatasetProvider(str, Enum):
    apify = "apify"
    local = "local_import"
