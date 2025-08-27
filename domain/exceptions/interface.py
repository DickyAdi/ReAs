from abc import ABC, abstractmethod

from .exception import DatabaseIntegrityError, DatabaseError


class DatabaseErrorMapperInterface(ABC):
    """Domain contract for database error mapper to translate database error to a specific domain exception error."""

    @classmethod
    @abstractmethod
    def map_error(cls, error) -> DatabaseError:
        """Implementation of database error mapper.

        Args:
            error (Any): Error thrown by database.

        Returns:
            BaseError: Specific domain exception error.
        """
        pass

    @classmethod
    @abstractmethod
    def map_integrity_error(cls, error) -> DatabaseIntegrityError:
        """Implementation of database integrity related error.

        Args:
            error (Any): Integrity related database error.

        Returns:
            DatabaseIntegrityError: Specific integrity domain exception error.
        """
        pass
