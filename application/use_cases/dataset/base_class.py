# from uuid import UUID
# from typing import Optional


# from application.predictor import PredictorApplication
# from application.extractor import ExtractorApplication
# from application.data_sources import DataSourceApplication
# from domain.uow import UnitOfWorkInterface
# from domain.predictor import PredictorInterface
# from domain.extractor import ExtractorInterface
# from domain.enums.datasets import DatasetProvider
# from domain.exceptions import DoubleIdentifierError, DatasetNotFoundError
# from domain.entities.reviews import ReviewEntity


# class BaseDatasetFlow:
#     def __init__(
#         self,
#         uow: UnitOfWorkInterface,
#         prediction_service: PredictorInterface,
#         extract_service: ExtractorInterface,
#     ):
#         self.uow = uow
#         self.predict_app = PredictorApplication(prediction_service)
#         self.extract_app = ExtractorApplication(extract_service)
#         self.data_source_app = DataSourceApplication(self.uow)

#     async def insert_to_dataset(
#         self,
#         reviews: list[ReviewEntity],
#         data_provider: DatasetProvider,
#         uow: UnitOfWorkInterface,
#         dataset_id: Optional[UUID] = None,
#         dataset_name: Optional[str] = None,
#         additional_metadata: Optional[dict] = None,
#     ):
#         if dataset_id and dataset_name:
#             raise DoubleIdentifierError(
#                 "Choose only 1 identifier", identifiers=["dataset_id", "dataset_name"]
#             )
#         # dataset = self.uow.datasets.get_dataset_by_id(id=dataset_id)
#         dataset = None
#         if dataset_id:
#             dataset = uow.datasets.get_dataset_by_id(id=dataset_id)
#         else:
#             dataset = uow.datasets.get_dataset_by_name(name=dataset_name)
#         if not dataset:
#             raise DatasetNotFoundError(identifier=dataset_id or dataset_name)
#         reviews_obj = await uow.reviews.create_review(reviews)
#         batch_stmt = await uow.reviews.batch_insert_statement(reviews_obj)
