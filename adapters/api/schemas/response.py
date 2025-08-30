from typing import Any
from fastapi.responses import JSONResponse
from fastapi import status


class BaseResponse(JSONResponse):
    def __init__(
        self, message: str = None, status_code: int = None, data: Any = None, **kwargs
    ):
        message = message or "Success"
        status_code = status_code or status.HTTP_200_OK

        super().__init__(
            status_code=status_code,
            content={"message": message, "data": data},
            **kwargs,
        )


class ResponseOk(BaseResponse):
    def __init__(self, message: str = None, data: Any = None, **kwargs):
        super().__init__(
            message=message, status_code=status.HTTP_200_OK, data=data, **kwargs
        )


class ResponseCreated(BaseResponse):
    def __init__(self, message: str = None, data: Any = None, **kwargs):
        message = message or "Resource created"
        super().__init__(
            message=message, data=data, status_code=status.HTTP_201_CREATED, **kwargs
        )
