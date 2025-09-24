from functools import wraps


def enforce_transaction(method):
    @wraps(method)
    async def wrapper(self, *args, **kwargs):
        if not self.uow.in_transaction:
            raise RuntimeError(
                f"Method `{method.__name__}` must be called inside a Unit of Work transaction"
            )
        return await method(self, *args, **kwargs)

    return wrapper
