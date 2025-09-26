from typing import Optional, Union

from domain.workers import WorkerInterface
from adapters.workers.pipeline import PipelineWorker


class DatasetPipeline(WorkerInterface):
    def __init__(
        self,
        # file: BinaryIO,
        data: Union[bytes, str],
        text_column: str,
        rating_column: Optional[str] = None,
        batch_size: Optional[int] = 512,
    ):
        self.worker = PipelineWorker(
            # file=file,
            data=data,
            text_column=text_column,
            rating_column=rating_column,
            batch_size=batch_size,
        )

    def run(self, **kwargs):
        task = self.worker.run_workflow(**kwargs)
        return task.id
