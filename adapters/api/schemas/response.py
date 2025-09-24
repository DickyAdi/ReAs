from typing import Any
from fastapi.responses import JSONResponse
from fastapi import status
from uuid import UUID
from datetime import datetime, date
from decimal import Decimal


class BaseResponse(JSONResponse):
    def __init__(
        self, message: str = None, status_code: int = None, data: Any = None, **kwargs
    ):
        message = message or "Success"
        status_code = status_code or status.HTTP_200_OK

        super().__init__(
            status_code=status_code,
            content={"message": message, "data": self.serialize(data)},
            **kwargs,
        )

    def serialize(self, data):
        if isinstance(data, list):
            return [self.serialize(o) for o in data]
        if hasattr(data, "to_dict"):
            return data.to_dict()
        if isinstance(data, dict):
            return {k: self.serialize(v) for k, v in data.items()}
        if isinstance(data, UUID):
            return str(data)
        if isinstance(data, datetime):
            return data.isoformat()
        if isinstance(data, date):
            return data.isoformat()
        if isinstance(data, Decimal):
            return float(data)
        return data


class ResponseOk(BaseResponse):
    def __init__(self, message: str = None, data: Any = None, **kwargs):
        super().__init__(
            message=message or "Ok", status_code=status.HTTP_200_OK, data=data, **kwargs
        )


class ResponseCreated(BaseResponse):
    def __init__(self, message: str = None, data: Any = None, **kwargs):
        message = message or "Resource created"
        super().__init__(
            message=message, data=data, status_code=status.HTTP_201_CREATED, **kwargs
        )


class ResponseAccepted(BaseResponse):
    def __init__(self, message: str = None, data=None, **kwargs):
        message = message or "Request accepted"
        super().__init__(
            message=message, status_code=status.HTTP_202_ACCEPTED, data=data, **kwargs
        )
