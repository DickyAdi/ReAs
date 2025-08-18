class UserNotFoundException(Exception):
    "Custom Exception when get user credentials returns None."
    def __init__(self, message="User not found."):
        self.message = message
        super().__init__(message)
    def __str__(self):
        return f"UserNotFoundException: {self.message}"

class InvalidResetPasswordToken(Exception):
    "Custom Exception when invalid reset password token met."
    def __init__(self, message="Reset token is invalid."):
        self.message = message
        super().__init__(message)
    def __str__(self):
        return f"InvalidResetPasswordToken: {self.message}"

class InvalidEmailVerifyToken(Exception):
    "Custom Exception when invalid email verify token met."

class EmailAlreadyValidated(Exception):
    "Custom Exception when user is already validated but askin for validation."

class UnpaidTransactions(Exception):
    "Custom Exception when user is asking for subscription status change but transaction status is not `paid`."
    def __init__(self, message="Transaction is unpaid."):
        self.message = message
        super().__init__(message)
    def __str__(self):
        return f"UnpaidTransactions: {self.message}"

class InvalidCredentials(Exception):
    "Custom Exception when given credentials is invalid."
    def __init__(self, message="Credentials is invalid."):
        self.message = message
        super().__init__(message)
    def __str__(self):
        return f"InvalidCredentials: {self.message}"
    
class InvalidAuthenticateToken(Exception):
    "Custom Exception when given authenticate token is invalid."
    def __init__(self, message="Authenticate token is invalid."):
        self.message = message
    def __str__(self):
        return f"InvalidAuthenticateToken: {self.message}"
    
class PlanRefError(Exception):
    "Custom Exception when new plan is not refer or based on existing plan."
    def __init__(self, message="New plan must be based on or refer to an existing plan."):
        self.message = message
        super().__init__(message)
    def __str__(self):
        return f"PlanRefError: {self.message}"

class PlanDeactivateError(Exception):
    "Custom Exception while trying to append/create new plan but unable to deactivate existing referer plan."
    def __init__(self, message="Deactivate referer plan was not finished."):
        self.message = message
        super().__init__(message)
    def __str__(self):
        return f"PlanDeactivateError: {self.message}"