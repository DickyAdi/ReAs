from enum import Enum


class DatasetStatus(str, Enum):
    # pending = "pending"
    # processing = "processing"
    # success = "success"
    # failed = "failed"
    # nores = "no result"
    # empty = "empty"
    # success = "success"
    # nores = "no result"
    pending = "pending"
    processing = "processing"
    extracted = "extracted"
    failed = "failed"


class DatasetProvider(str, Enum):
    scrape = "scrape"
    local = "local import"
