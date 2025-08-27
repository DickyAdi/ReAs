from sqlalchemy.exc import (
    IntegrityError,
    InterfaceError,
    OperationalError,
    SQLAlchemyError,
)
from psycopg.errors import (
    UniqueViolation,
    ForeignKeyViolation,
    NotNullViolation,
    CheckViolation,
)
import re

from domain.exceptions.entity import ParsedConstraintData, PatternDictType
from domain.exceptions.interface import DatabaseErrorMapperInterface
from domain.exceptions.exception import (
    DatabaseError,
    DatabaseIntegrityError,
    DatabaseInterfaceError,
    DatabaseOperationalError,
    CheckConstraintViolationError,
    UniqueConstraintViolationError,
    NotNullConstraintViolationError,
    ForeignKeyViolationError,
)


class DatabaseErrorMapper(DatabaseErrorMapperInterface):
    PATTERNS: PatternDictType = {
        "constraint_name": re.compile(r'constraint "([^"]+)"'),
        "field_name": re.compile(r"Key \(([^)]+)\)"),
        "fk_value": re.compile(r"Key \({field_name}\)=\((.*?)\) is not present"),
        "fk_orig_table": re.compile(r'insert or update on table "([^"]+)'),
        "fk_ref_table": re.compile(r'is not present in table "([^"]+)'),
        "unique_value": re.compile(r"Key \({field_name}\)=\((.*?)\) already exists"),
        "check_value": re.compile(r"Failing row contains \((.*?)\)"),
        "check_table": re.compile(r'new row for relation "([^"]+)'),
        "not_null_value": re.compile(r"Failing row contains \((.*?)\)"),
        "not_null_column": re.compile(r'null value in column "([^"]+)'),
    }

    @classmethod
    def map_error(cls, error: SQLAlchemyError) -> DatabaseError:
        """Map SQLAlchemy error to domain exception error.

        Args:
            error (SQLAlchemyError): Actual error thrown by database.

        Returns:
            DatabaseError: Specific domain exception error.
        """
        error_handler = {
            IntegrityError: cls.map_integrity_error,
            InterfaceError: cls._handle_interface_error,
            OperationalError: cls._handle_operational_error,
        }
        handler = error_handler.get(type(error), cls._handle_unknown_error)
        return handler(error)

    @classmethod
    def map_integrity_error(cls, error: IntegrityError) -> DatabaseIntegrityError:
        """Map SQLAlchemy IntegrityError to Domain exception error.

        Args:
            error (IntegrityError): Actual error thrown by database.

        Returns:
            DatabaseIntegrityError: Specific domain exception error.
        """
        orig_err = error.orig
        error_message = str(orig_err)

        error_handlers = {
            UniqueViolation: cls._handle_unique_error,
            ForeignKeyViolation: cls._handle_fk_error,
            NotNullViolation: cls._handle_not_null_error,
            CheckViolation: cls._handle_check_error,
        }
        handler = error_handlers.get(
            type(orig_err), cls._handle_unknown_integrity_error
        )
        return handler(error_message)

    @classmethod
    def _extract_constraint_data(cls, error_message: str) -> ParsedConstraintData:
        """Extract base constraint data such as constraint name and field name.

        Args:
            error_message (str): Error message thrown by database.

        Returns:
            ParsedConstraintData: Parsed DTO to be modify further.
        """
        data = ParsedConstraintData()

        if match := cls.PATTERNS["constraint_name"].search(error_message):
            data.name = match.group(1)

        if match := cls.PATTERNS["field_name"].search(error_message):
            data.field_name = match.group(1)

        return data

    @classmethod
    def _handle_interface_error(cls, error: InterfaceError) -> DatabaseInterfaceError:
        """Translate SQLALchemy `InterfaceError` to `DatabaseInterfaceError`.

        Args:
            error (InterfaceError): Actual `InterfaceError` thrown by database.

        Returns:
            DatabaseInterfaceError: Domain database interface error.
        """
        return DatabaseInterfaceError(
            message="Adapter could not finish a task", raw_message=str(error)
        )

    @classmethod
    def _handle_operational_error(
        cls, error: OperationalError
    ) -> DatabaseOperationalError:
        """Translate SQLAlchemy `OperationalError` to `DatabaseOperationalError`.

        Args:
            error (OperationalError): Actual `OperationalError` thrown by database.

        Returns:
            DatabaseOperationalError: Domain database operational error.
        """
        return DatabaseOperationalError(
            message="Database unreachable", raw_message=str(error)
        )

    @classmethod
    def _handle_unique_error(cls, error_message: str) -> UniqueConstraintViolationError:
        """Handle unique constraint violation.

        Args:
            error_message (str): Error message thrown by database.

        Returns:
            UniqueConstraintViolationError: Domain unique constraint violation error.
        """
        data = cls._extract_constraint_data(error_message=error_message)

        if data.field_name:
            formatted_value_pattern = re.compile(
                cls.PATTERNS["unique_value"].pattern.format(
                    field_name=re.escape(data.field_name)
                )
            )
            if match := formatted_value_pattern.search(error_message):
                data.field_value = match.group(1)
        return UniqueConstraintViolationError(
            field_name=data.field_name,
            field_value=data.field_value,
            constraint_name=data.name,
        )

    @classmethod
    def _handle_fk_error(cls, error_message: str) -> ForeignKeyViolationError:
        """Handle foreign key constraint violation error.

        Args:
            error_message (str): Error message thrown by database.

        Returns:
            ForeignKeyViolationError: Domain foreign key constraint violation error.
        """
        data = cls._extract_constraint_data(error_message)

        if match := cls.PATTERNS["fk_orig_table"].search(error_message):
            data.origin_table = match.group(1)

        if match := cls.PATTERNS["fk_ref_table"].search(error_message):
            data.referenced_table = match.group(1)

        if data.field_name:
            formatted_value_pattern = re.compile(
                cls.PATTERNS["fk_value"].pattern.format(
                    field_name=re.escape(data.field_name)
                )
            )
            if match := formatted_value_pattern.search(error_message):
                data.field_value = match.group(1)

        return ForeignKeyViolationError(
            referenced_table=data.referenced_table,
            origin_table=data.origin_table,
            referenced_name=data.field_name,
            referenced_value=data.field_value,
            constraint_name=data.name,
        )

    @classmethod
    def _handle_check_error(cls, error_message: str) -> CheckConstraintViolationError:
        """Handle check constraint violation error.

        Args:
            error_message (str): Actual error message thrown by database.

        Returns:
            CheckConstraintViolationError: Domain check constraint violation error.
        """
        data = ParsedConstraintData()

        if match := cls.PATTERNS["constraint_name"].search(error_message):
            data.name = match.group(1)

        if match := cls.PATTERNS["check_table"].search(error_message):
            data.table_name = match.group(1)

        if match := cls.PATTERNS["check_value"].search(error_message):
            data.field_value = match.group(1)

        message = (
            f"Check constraint `{data.name}` failed in table `{data.table_name}` "
            f"with value `{data.field_value}`"
        )

        return CheckConstraintViolationError(constraint_name=data.name, message=message)

    @classmethod
    def _handle_not_null_error(
        cls, error_message: str
    ) -> NotNullConstraintViolationError:
        """Handle not null constraint violation error.

        Args:
            error_message (str): Actual error message thrown by database.

        Returns:
            NotNullConstraintViolationError: Domain not null constraint violation error
        """
        data = ParsedConstraintData()

        if match := cls.PATTERNS["not_null_column"].search(error_message):
            data.field_name = match.group(1)

        if match := cls.PATTERNS["not_null_value"].search(error_message):
            data.field_value = match.group(1)

        return NotNullConstraintViolationError(
            column_name=data.field_name, value=data.field_value
        )

    @classmethod
    def _handle_unknown_integrity_error(
        cls, error_message: str
    ) -> DatabaseIntegrityError:
        """Handle unknown/unexpected integrity error.

        Args:
            error_message (str): Actual error message thrown by database.

        Returns:
            DatabaseIntegrityError: Domain database integrity error for unknown integrity error.
        """
        return DatabaseIntegrityError(
            message="Unknown integrity error",
            error_code="UNKNOWN_INTEGRITY_VIOLATION",
            details={"original_message": error_message},
        )

    @classmethod
    def _handle_unknown_error(cls, error: SQLAlchemyError) -> DatabaseError:
        """Handle unknown database error.

        Args:
            error (SQLAlchemyError): Actual error thrown by database.

        Returns:
            DatabaseError: Domain database error for unknown error.
        """
        return DatabaseError(
            message="Unknown database error",
            error_code="UNKNOWN_DATABASE_ERROR",
            details={"original_message": str(error)},
        )
