from celery import Celery
from celery.signals import task_failure

from config.settings import settings
from loggers.log import get_loggers

from application.use_cases.prediction import PredictionUseCase
from infrastructure.registry.model import ModelRegistry
from infrastructure.predictor import PredictionService

PREDICT_UC = None
logger = get_loggers("reas.worker.celery")


@task_failure.connect
def task_failure_handler(
    sender=None,
    task_id=None,
    exception=None,
    args=None,
    kwargs=None,
    traceback=None,
    einfo=None,
    **kw,
):
    logger.error(
        f"Task {sender.name} - [{task_id}] failed: {exception}",
        exc_info=einfo.exc_info,
        extra={
            "task_name": sender.name,
            "task_id": task_id,
            "args": args,
            "kwargs": kwargs,
            "exception": str(exception),
        },
    )


def get_predict_uc():
    global PREDICT_UC
    if PREDICT_UC is None:
        PREDICT_UC = PredictionUseCase(
            model_registry=ModelRegistry(), predict_service=PredictionService()
        )
    return PREDICT_UC


app = Celery(
    "reas",
    broker=str(settings.redis_url),
    backend=str(settings.redis_url),
    include=["adapters.workers.tasks.dummy", "adapters.workers.pipeline.pipeline"],
)

if __name__ == "__main__":
    app.start()
