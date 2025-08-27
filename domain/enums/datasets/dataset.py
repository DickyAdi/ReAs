from enum import Enum


class DatasetStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    success = "success"
    failed = "failed"
    nores = "no result"


class DatasetProvider(str, Enum):
    scrape = "scrape"
    local = "local import"
