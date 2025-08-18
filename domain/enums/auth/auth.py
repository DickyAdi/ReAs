from enum import Enum

class TokenPurpose(str, Enum):
    authenticate = 'authenticate'
    verify_email = 'email_verification'
    forget_password = 'forget_password'