from ..main import app


@app.task
def dummy_add(x, y):
    return x + y
