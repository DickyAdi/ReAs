from typing import Optional, Union
from uuid import UUID

# from .i.services import PipelineService
# from .internals.services import PipelineService
from .internals.task import distribute_inference, scrape_gmaps, rollback_pipeline
from domain.enums.texts import TextLanguage


class PipelineWorker:
    def __init__(
        self,
        # file: BinaryIO,
        data: Union[bytes, str],
        text_column: str,
        rating_column: Optional[str] = None,
        batch_size: Optional[int] = 512,
    ):
        self.data = data
        self.text_column = text_column
        self.rating_column = rating_column
        self.batch_size = batch_size

    def run_workflow(
        self,
        dataset_id: UUID,
        provider_ref: UUID,
        language: TextLanguage,
        model_name: Optional[str] = "default",
    ):
        if isinstance(self.data, str):
            return (
                (
                    scrape_gmaps.s(resource_url=self.data)
                    | distribute_inference.s(
                        text_column="None",
                        dataset_id=dataset_id,
                        provider_ref=provider_ref,
                        language=language,
                        rating_column="None",
                        model_name=model_name,
                        batch_size=self.batch_size,
                    )
                )
                .on_error(
                    rollback_pipeline.s(
                        dataset_id=dataset_id, provider_ref=provider_ref
                    )
                )
                .delay()
            )
        else:
            return (
                distribute_inference.s(
                    data=self.data,
                    text_column=self.text_column,
                    rating_column=self.rating_column,
                    dataset_id=dataset_id,
                    provider_ref=provider_ref,
                    language=language,
                    model_name=model_name,
                    batch_size=self.batch_size,
                )
                .on_error(
                    rollback_pipeline.s(
                        dataset_id=dataset_id, provider_ref=provider_ref
                    )
                )
                .delay()
            )
