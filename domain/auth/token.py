from dataclasses import dataclass

@dataclass
class Token:
    access_token:str
    token_type:str

@dataclass
class TokenData:
    email:str | None = None