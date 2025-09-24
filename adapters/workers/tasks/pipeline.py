from ..main import app


@app.task
def batch_inference(reviews: list[str]): ...
