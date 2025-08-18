from datetime import datetime, timedelta, timezone

from domain.auth import AuthInterface
from domain.enums.auth import TokenPurpose


class AuthApplication:
    """Authentication application for authentication services."""

    def __init__(self, service: AuthInterface):
        self.service = service

    def decode_token(self, token):
        """Decode token based on the given token.

        Args:
            token (str): JWT token

        Returns:
            dict: Decoded JWT token
        """
        payload = self.service.decode_token(token=token)
        return payload

    def verify_password(self, user_password: str, db_password: str):
        """Verify password function between user input and database value

        Args:
            user_password (str): User input password value.
            db_password (str): Password value stored in database.

        Returns:
            bool: True if password match, False if password doesn't match.
        """
        return self.service.verify_password(
            user_password=user_password, db_password=db_password
        )

    def create_token(
        self,
        email: str,
        token_version: int,
        expire_time: timedelta,
        purpose: TokenPurpose,
    ):
        """Create JWT token based on the given parameter. This function can be used for different kind of purposes that supported by `TokenPurpose` Enums.

        Args:
            email (str): Subject email that holds this token.
            token_version (int): Token version for this token.
            expire_time (timedelta): Exp time for this token.
            purpose (TokenPurpose): Token purpose of this token.

        Returns:
            str: Created token.
        """
        exp = datetime.now(timezone.utc) + expire_time
        token_data = {
            "sub": email,
            "exp": exp,
            "token_version": token_version,
            "purpose": purpose.name,
        }
        token = self.service.create_access_token(data=token_data)
        return token

    def hash_password(self, value: str):
        """Utility function to hash user input password value.

        Args:
            value (str): User input password value.

        Returns:
            str: Hashed password value.
        """
        return self.service.hash(value=value)
