import asyncio
import threading

_loop_local = threading.local()


def get_loop():
    if not hasattr(_loop_local, "loop"):
        _loop_local.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop_local.loop)
    return _loop_local.loop


def run_sync(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        raise RuntimeError(
            "Calling run_sync inside async event loop, use await instead."
        )
    loop = get_loop()
    return loop.run_until_complete(coro)
