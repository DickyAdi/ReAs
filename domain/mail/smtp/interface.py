from abc import ABC, abstractmethod

from typing import TypedDict

class SMTP_CREDENTIALS(TypedDict):
    smtp_host:str
    smtp_username:str
    smtp_password:str

class SmtpInterface(ABC):
    @abstractmethod
    async def send(self, email, sender, recepients, smtp_credentials:SMTP_CREDENTIALS):
        pass