from dataclasses import dataclass
from typing import Optional, TypedDict
from re import Pattern


@dataclass
class ParsedConstraintData:
    name: Optional[str] = None
    field_name: Optional[str] = None
    field_value: Optional[str] = None
    table_name: Optional[str] = None
    origin_table: Optional[str] = None
    referenced_table: Optional[str] = None


class PatternDictType(TypedDict):
    constraint_name: Pattern
    field_name: Pattern
    fk_value: Pattern
    fk_orig_table: Pattern
    fk_ref_table: Pattern
    unique_value: Pattern
    check_value: Pattern
    check_table: Pattern
    not_null_value: Pattern
    not_null_column: Pattern
