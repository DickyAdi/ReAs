from typing import Optional, Union
from uuid import UUID
from domain.enums.auth import TokenPurpose


class BaseError(Exception):
    """Base exception for all domain errors."""

    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __str__(self):
        return f"{self.__class__.__name__}: {self.message}"

    def __reduce__(self):
        return (
            self.__class__,
            (
                getattr(self, "message", "Base exception message"),
                getattr(self, "error_code", None),
                getattr(self, "details", {}),
            ),
        )


# Non-specific error.
class PaginationTypeError(BaseError):
    """Raised when pagination could not be done due to type missmatch in `offset` or `limit`."""

    def __init__(self, message: str = None, offset: int = None, limit: int = None):
        """Initializing error for PaginationTypeError due to type missmatch in `offset` or `limit`

        Args:
            offset (_type_): Offset value
            limit (_type_): Limit value
        """
        message = message or "Invalid value for `offset` or `limit`"
        details = {}
        if limit:
            details["limit"] = limit
        if offset:
            details["offset"] = offset
        super().__init__(
            message=message, error_code="PAGINATION_TYPE_ERROR", details=details
        )


class DoubleIdentifierError(BaseError):
    """Raised when double identifier passed to a function/method that expect to receive only 1 identifier."""

    def __init__(self, message: str = None, identifiers: list[str] = None):
        """Initializing error for DoubleIdentifierError.

        Args:
            message (str, optional): Message that will be returned as exception. Defaults to None.
            identifiers (list, optional): List of available identifiers. Defaults to None.
        """
        message = message or "Choose only one identifier."
        details = {}
        if identifiers:
            details["available_identifiers"] = identifiers
        super().__init__(
            message=message, error_code="DOUBLE_IDENTIFIER_ERROR", details=details
        )


class RequestRateLimitedError(BaseError):
    """Raised when request surpass limit"""

    def __init__(self, limit: int = 10, window: int = 60):
        message = "Too many request"
        details = {"limit": limit, "window": window}
        super().__init__(
            message=message, error_code="RATE_LIMIT_ERROR", details=details
        )


# User related error
class EmailAlreadyExistError(BaseError):
    """Raised when registering with existing email in the system."""

    def __init__(self, email: str):
        message = f"Email {email} already exists"
        details = {"email": email}
        super().__init__(
            message=message, error_code="EMAIL_ALREADY_EXISTS_ERROR", details=details
        )


class UserNameAlreadyExistError(BaseError):
    """Raised when name already exists in the system."""

    def __init__(self, name: str):
        message = f"Name `{name}` already exists"
        details = {"name": name}
        super().__init__(
            message=message,
            error_code="USER_NAME_ALREADY_EXISTS_ERROR",
            details=details,
        )


# Dataset related error
class DatasetNameAlreadyExistError(BaseError):
    """Raised when dataset name already exists in the system."""

    def __init__(self, name: str):
        message = f"Dataset with name `{name}` already exists"
        details = {"name": name}
        super().__init__(
            message=message,
            error_code="DATASET_NAME_ALREADY_EXISTS_ERROR",
            details=details,
        )


class DatasetNotFoundError(BaseError):
    """Raised when dataset doesnt exists in the system."""

    def __init__(self, identifier: Optional[Union[str, UUID]] = None):
        message = (
            f"Dataset with identifier `{str(identifier)}` doesnt exists in the system"
            or "Dataset doesnt exists"
        )
        details = {}
        if identifier:
            details = {"identifier": str(identifier)}
        super().__init__(
            message=message, error_code="DATASET_NOT_FOUND", details=details
        )


# Data source related error
class DataSourceNotFoundError(BaseError):
    "Raised when data source doesnt exists in the system."

    def __init__(self, identifier: Optional[Union[str, UUID]] = None):
        message = (
            f"Data source with identifier `{str(identifier)}` doesnt exists in the system"
            or "Data source doesnt exists"
        )
        details = {}
        if identifier:
            details = {"identifier": str(identifier)}
        super().__init__(
            message=message, error_code="DATA_SOURCE_NOT_FOUND", details=details
        )


# Extraction related error
class NotEnoughReviewsError(BaseError):
    "Raised when reviews less than the desired number"

    def __init__(self, n: int):
        message = (
            "Not enough number of reviews to be extracted. Must at least provide 2 reviews"
            or f"Not enough number of reviews to be extracted. Must at least provide 2 reviews. Got {n}"
        )
        details = {}
        if n:
            details["number_of_reviews"] = n
        super().__init__(
            message=message, error_code="NOT_ENOUGH_REVIEW", details=details
        )


# Authentication and authorization.
class UserNotFoundError(BaseError):
    """Raised when a user cannot be found in the system."""

    def __init__(self, user_identifier: str = None):
        """Initializing error for UserNotFoundError.

        Args:
            user_identifier (str, optional): User identifier that does not found within the system. Defaults to None.
        """
        message = "User not found"
        details = {}
        if user_identifier:
            message = f"User with identifier `{user_identifier}` not found."
            details["user_identifier"] = user_identifier
        super().__init__(message=message, error_code="USER_NOT_FOUND", details=details)


class InvalidCredentialsError(BaseError):
    """Raised when provided authentication credentials are invalid."""

    def __init__(self, credentials_type: str = "login"):
        """Initializing error for InvalidCredentialsError.

        Args:
            credentials_type (str, optional): Needed valid credentials type. Defaults to "login".
        """
        message = f"Invalid {credentials_type} "
        super().__init__(
            message=message,
            error_code="INVALID_CREDENTIALS_ERROR",
            details={"credentials_type": credentials_type},
        )


class InvalidTokenError(BaseError):
    """Base class for token-related error."""

    def __init__(self, token_type: TokenPurpose, message: str = None):
        message = message or f"Invalid {token_type.value} token."
        super().__init__(
            message=message,
            error_code=f"INVALID_{token_type.value.upper()}_TOKEN_ERROR",
            details={"token_type": token_type},
        )


class InvalidResetPasswordTokenError(InvalidTokenError):
    """Raised when provided reset token is invalid."""

    def __init__(self, message: str = None):
        super().__init__(token_type=TokenPurpose.forget_password, message=message)


class InvalidEmailVerificationTokenError(InvalidTokenError):
    """Raised when provided email verification token is invalid."""

    def __init__(self, message: str = None):
        super().__init__(token_type=TokenPurpose.verify_email, message=message)


class InvalidAuthenticateTokenError(InvalidTokenError):
    """Raised when provided authenticate token is invalid."""

    def __init__(self, message: str = None):
        super().__init__(token_type=TokenPurpose.authenticate, message=message)


class InsufficientRoleError(BaseError):
    """Raised when role is insufficient to access protected source."""

    def __init__(self, role: str, minimum_role: str = None):
        message = (
            f"Insufficient role to access protected source with minimum role `{minimum_role}`"
            if minimum_role
            else "Insufficient role to access protected source"
        )
        details = {
            "role": role,
        }
        if minimum_role:
            details["minimum_role"] = minimum_role
        super().__init__(
            message=message, error_code="INSUFFICIENT_ROLE_ERROR", details=details
        )


class InsufficientTierError(BaseError):
    """Raised when tier is insufficient to access protected source."""

    def __init__(self, tier: str, minimum_tier: str = None):
        message = (
            f"Insufficient tier to access protected source with minimum tier `{minimum_tier}`"
            if minimum_tier
            else "Insufficient tier to access protected source"
        )
        details = {"tier": tier}
        if minimum_tier:
            details["minimum_tier"] = minimum_tier
        super().__init__(
            message=message, error_code="INSUFFICIENT_TIER_ERROR", details=details
        )


# SMTP related error
class FailedToSendEmail(BaseError):
    """Raised when failed to send email to users email."""

    def __init__(
        self, users_email: str, smtp_provider: str = None, smtp_error: str = None
    ):
        message = f"Failed to send email to `{users_email}`"
        details = {}
        if smtp_provider:
            details["smtp_provider"] = smtp_provider
        if smtp_error:
            details["traceback"] = smtp_error
        super().__init__(
            message=message, error_code="FAILED_TO_SEND_EMAIL", details=details
        )


# Email verifications error
class EmailAlreadyVerifiedError(BaseError):
    """Raised when users email is already verified when asking for verification."""

    def __init__(self, email: str = None):
        message = "Email is already verified."
        details = {}
        if email:
            details["email"] = email
        super().__init__(
            message=message, error_code="EMAIL_ALREADY_VERIFIED", details=details
        )


# Subscriptions, payments, and transactions error.
class UnpaidTransactionsError(BaseError):
    """Raised when attempting to process a subscription with unpaid transactions."""

    def __init__(self, transaction_id: str = None):
        """Initiating error when attempting to process a subscription with unpaid transaction.

        Args:
            transaction_id (str, optional): Transactions_id. Defaults to None.
        """
        message = "Transaction must be paid before processing subscription."
        details = {}
        if transaction_id:
            details["transaction_id"] = transaction_id
        super().__init__(
            message=message, error_code="TRANSACTION_UNPAID", details=details
        )


class PlanReferenceError(BaseError):
    """Raised when a subscription plan reference is invalid."""

    def __init__(self, plan_code: str):
        """Initialize plan reference error when trying to create new plan but old plan doesnt exists.

        Args:
            plan_code (str): Plan code that are tried to be created.
        """
        message = "New subscription plan must reference an existing active plan."
        details = {"plan_code": plan_code}
        super().__init__(
            message=message, error_code="INVALID_PLAN_REFERENCE", details=details
        )


class PlanDeactivateError(BaseError):
    """Raised when encounter an error while deactivating old plan when creating new plan."""

    def __init__(self, plan_code: str):
        """Initialize plan deactivation error when creating new plan but deactivating old plan encounter an error, thus the old plan is not deactivated.

        Args:
            plan_code (str): Plan code that are tried to be deactivated.
        """
        message = "Error while deactivating old plan."
        details = {"plan_code": plan_code}
        super().__init__(
            message=message, error_code="PLAN_DEACTIVATION_FAILED", details=details
        )


# IO error
class ColumnNotFoundError(BaseError):
    """Raised when required column is not found in CSV data."""

    def __init__(self, text_column: str, available_column: list[str] = None):
        """Initialize column not found error when trying to read a required column but not found in the CSV data.

        Args:
            text_column (str): Required column.
            available_column (list[str], optional): Available column in the CSV data. Defaults to None.
        """
        message = f"Column `{text_column}` not found in CSV data"
        details = {"required_column": text_column, "available_column": available_column}
        super().__init__(
            message=message, error_code="COLUMN_NOT_FOUND", details=details
        )


class InvalidEncodingError(BaseError):
    """Raised when CSV data has incorrect encoding."""

    def __init__(self, expected_encoding: str = "utf-8", received_encoding: str = None):
        """Initialize invalid encoding error when trying to decode CSV data but encounter UnicodeDecodeException.

        Args:
            expected_encoding (str, optional): Expected encoding a file must have. Defaults to 'utf-8'.
            received_encoding (str, optional): Received file encoding type. Defaults to None.
        """
        message = f"CSV must be encoded in `{expected_encoding}`"
        details = {"expected_encoding": expected_encoding}
        if received_encoding:
            details["received_encoding"] = received_encoding
        super().__init__(
            message=message, error_code="INVALID_ENCODING", details=details
        )


class UnsupportedFileType(BaseError):
    """Raised when provided with unsupported file types."""

    def __init__(self, expected_type: str = "CSV", received_type: str = None):
        message = (
            f"Expecting type {expected_type} received type {received_type}"
            if expected_type and received_type
            else "Unsupported file type."
        )
        details = {}
        if expected_type:
            details["expected_type"] = expected_type
        if received_type:
            details["received_type"] = received_type
        super().__init__(
            message=message, error_code="UNSUPPORTED_FILE", details=details
        )


class FileTooLarge(BaseError):
    """Raised when provided files are too large."""

    def __init__(self, max_size: int):
        """Initializing file too large exception, used when received files exceeds limit.

        Args:
            max_size (int): Received file size in MB (megabyte).
        """
        message = f"File size exceeds {max_size} MB limit"
        details = {"max_size": max_size}
        super().__init__(message=message, error_code="FILE_TOO_LARGE", details=details)


# Scraping error
class UnsupportedDomainToScrape(BaseError):
    """Raised when user pass unsupported domain for scraping."""

    def __init__(self, platform: str, message: str = None):
        message = message or f"Ensure the given url match the expected {platform} url"
        details = {"platform": platform}
        super().__init__(
            message=message, error_code="UNSUPPORTED_DOMAIN_TO_SCRAPE", details=details
        )


# Security error
class SecurityError(BaseError):
    """Raised when suspicious behavior detected."""

    def __init__(self, message: str = None, **kwargs):
        """Initializing security error, used when facing suspicious behavior.

        Args:
            message (str): message to be raised.
            **kwargs (Any): Additional details. See usage.

        Usages:
        ```python
            >>> raise SecurityError(message='Something went wrong', attack_vector='query path')
            >>> raise SecurityError(sus_value='some value')
        ```
        """
        message = (
            message or "Due to security concerns, we could not process your request"
        )
        details = {}
        if kwargs:
            for key, value in kwargs.items():
                details[key] = value
        super().__init__(
            message=message, error_code="SECURITY_BREACH_POTENTIAL", details=details
        )


# Database error
class DatabaseError(BaseError):
    """Base class for database error."""

    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(
            message=message,
            error_code=error_code or "DATABASE_ERROR",
            details=details or {},
        )


class DatabaseIntegrityError(DatabaseError):
    """Base class for database integrity exceptions."""

    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(
            message=message,
            error_code=error_code or "DATABASE_INTEGRITY_ERROR",
            details=details or {},
        )


class DatabaseInterfaceError(DatabaseError):
    """Raised when database adapter fail to finish a task."""

    def __init__(self, message: str, raw_message: str = None):
        super().__init__(
            message=message or "Database adapter could not finish a task",
            error_code="DATABASE_ADAPTER_ERROR",
            details={"original_message": raw_message} or {},
        )


class DatabaseOperationalError(DatabaseError):
    """Raised when database is unreachable."""

    def __init__(self, message: str, raw_message: str = None):
        super().__init__(
            message=message or "Database is unreachable",
            error_code="DATABASE_UNREACHABLE",
            details={"original_message": raw_message} or {},
        )


class UniqueConstraintViolationError(DatabaseIntegrityError):
    """Raise when unique constraint is violated."""

    def __init__(self, field_name: str, field_value: str, constraint_name: str = None):
        message = f"`{field_name}` : `{field_value}` already exists"
        details = {
            "field_name": field_name,
            "field_value": field_value,
            "constraint_name": constraint_name or "unique_constraint",
        }
        super().__init__(
            message=message, error_code="UNIQUE_CONSTRAINT_VIOLATION", details=details
        )


class ForeignKeyViolationError(DatabaseIntegrityError):
    """Raised when foreign key constraint is violated"""

    def __init__(
        self,
        referenced_table: str,
        origin_table: str,
        referenced_name: str,
        referenced_value: str,
        constraint_name: str = None,
    ):
        message = f"On table `{origin_table}` cannot refer to {referenced_name} with value of `{referenced_value}` on table `{referenced_table}`"
        details = {
            "referenced_column": referenced_name,
            "referenced_value": referenced_value,
            "referenced_table": referenced_table,
            "origin_table": origin_table,
            "constraint_name": constraint_name or "foreign_key_constraint",
        }
        super().__init__(
            message=message, error_code="FOREIGN_KEY_VIOLATION", details=details
        )


class CheckConstraintViolationError(DatabaseIntegrityError):
    """Raised when check constraint is violated."""

    def __init__(self, constraint_name: str, message: str = None):
        message = message or "Check constraint is violated"
        details = {"constraint_name": constraint_name}
        super().__init__(
            message=message,
            error_code="CHECK_CONSTRAINT_VIOLATION",
            details=details,
        )


class NotNullConstraintViolationError(DatabaseIntegrityError):
    """Raised when certain column is null where it is non-nullable."""

    def __init__(self, column_name: str, value: str = None):
        message = f"Null value in non-nullable column `{column_name}`"
        details = {"column_name": column_name}
        if value:
            details["value"] = value
        super().__init__(
            message=message, error_code="NOT_NULL_CONSTRAINT_VIOLATION", details=details
        )
