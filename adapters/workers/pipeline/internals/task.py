from itertools import chain
from typing import Optional, Union
import requests
import ijson
from datetime import datetime, timezone
from uuid import UUID
from celery import chord, Task
import io

from ...main import app, get_predict_uc
from ...database import SessionFactory
from ...utils import run_sync

from domain.enums.texts import TextLanguage, TextPlatform
from domain.enums.datasets import DatasetStatus
from application.reviews import ReviewApplication
from application.datasets import DatasetApplication
from application.data_sources import DataSourceApplication
from application.insights import InsightApplication
from application.use_cases.extraction import ExtractionUseCase
from application.scraping import ScrapingApplication
from application.io import JsonStreamerApplication, ReviewCsvStreamerApplication


from infrastructure.extractor import ExtractionService
from infrastructure.externals.outscraper import ScrapingService
from infrastructure.io.json_streamer import JsonStreamer
from infrastructure.io.csv_review_streamer import ReviewCsvStreamer


@app.task(name="extract-scrape_gmaps")
def scrape_gmaps(resource_url: str, query_params: dict):
    data = ScrapingApplication(service=ScrapingService()).build_hit(
        resource_url=resource_url, query_params=query_params
    )
    req_id = data.json().get("id")
    if not req_id:
        raise ValueError("Response doesnt contain id.")
    return req_id


# * Due to celery pickleable, for now we cannot use polling in a separate task to return requests http response.
# * The workaround is putting the polling within the distribute_inference tasks.
# * Overall the flow for scrape is [user asks scrape -> send scrape request and pass the id to distribute_inference -> poll the results within the task]
# * later migrate to MinIO if possible


def poll_scrape_result(req_id):
    data = requests.get(f"https://api.outscraper.cloud/requests/{req_id}", stream=True)
    parser = ijson.parse(data.raw)
    status = None
    for prefix, event, value in parser:
        if prefix == "status" and event == "string":
            status = value
            break
    if status == "Success":
        return data.raw
    return None


@app.task(
    bind=True,
    name="extract-distribute_inference",
    max_retries=20,
    default_retry_delay=60,
)
def distribute_inference(
    self: Task,
    data: Union[bytes, str],
    text_column: str,
    dataset_id: UUID,
    provider_ref: UUID,
    language: TextLanguage,
    rating_column: Optional[str],
    model_name: Optional[str] = "default",
    batch_size: Optional[int] = 512,
):
    inference_tasks = []
    time_now = datetime.now(timezone.utc)
    if isinstance(data, str):
        data = poll_scrape_result(data)
        if not data:
            raise self.retry()
        streamer = JsonStreamerApplication(
            service=JsonStreamer(raw_json=data, batch_size=batch_size)
        ).get_streamer()
        with streamer(prefix="data.item.reviews_data.item") as s:
            while item := s.next_batch():
                inference_tasks.append(
                    batch_inference.s(
                        reviews=item,
                        dataset_id=dataset_id,
                        provider_ref=provider_ref,
                        platform=TextPlatform.gmaps,
                        language=language,
                        post_date=time_now,
                        model_name=model_name,
                    )
                )
    else:
        streamer = ReviewCsvStreamerApplication(
            service=ReviewCsvStreamer(
                file=io.BytesIO(data),
                text_column=text_column,
                rating_column=rating_column,
                batch_size=batch_size,
            )
        ).get_streamer()
        with streamer as s:
            while item := s.next_batch():
                inference_tasks.append(
                    batch_inference.s(
                        reviews=item,
                        dataset_id=dataset_id,
                        provider_ref=provider_ref,
                        platform=TextPlatform.user,
                        language=language,
                        post_date=time_now,
                        model_name=model_name,
                    )
                )
    if not inference_tasks:
        raise ValueError("No tasks available")
    workflow = (
        chord(
            inference_tasks,
            (
                insert_reviews_to_db.s()
                | extract_insight.s(language="indonesian")
                | upsert_insights.s()
                | finish_extract.s()
            ),
        )
        .on_error(rollback_pipeline.s(dataset_id=dataset_id, provider_ref=provider_ref))
        .delay()
    )
    return workflow


@app.task(name="extract-batch_inference")
def batch_inference(
    reviews: list[dict],
    platform: TextPlatform,
    dataset_id: UUID,
    provider_ref: UUID,
    language: TextLanguage,
    post_date: datetime,
    model_name: Optional[str] = "default",
) -> list[dict]:
    predict_uc = get_predict_uc()
    cleaned = predict_uc.clean(reviews)
    cleaned_text = [text.get("text") for text in cleaned]
    predicted_sentiment = predict_uc.predict(texts=cleaned_text, model_name=model_name)
    assert len(cleaned_text) == len(predicted_sentiment), (
        f"Len doesnt match. Got {len(cleaned_text)} and {len(predicted_sentiment)}"
    )
    predicted = []
    for raw_dict, sentiment in zip(cleaned, predicted_sentiment):
        if not raw_dict.get("text") == sentiment.get("text"):
            raise ValueError("Text doesnt match, aborting processing further")
        predicted.append(
            {
                "text": raw_dict.get("text"),
                "rating": raw_dict.get("rating", None),
                "sentiment": sentiment.get("sentiment"),
                "platform": platform,
                "language": language,
                "provider_ref": provider_ref,
                "dataset_id": dataset_id,
                "post_date": p_date
                if (p_date := raw_dict.get("post_date", None)) is not None
                else post_date,
            }
        )
    return predicted


@app.task(name="extract-insert_reviews_to_db")
def insert_reviews_to_db(reviews: list[list[dict]]):
    async def job(inside_reviews):
        uow = SessionFactory.create()
        async with uow as u:
            review_app = ReviewApplication(uow=u)
            dataset_id = await review_app.insert_reviews_from_dict(
                reviews=inside_reviews, get_dataset_id=True, generate_defaults=True
            )
            await u.commit()
            return dataset_id

    all_reviews = list(chain.from_iterable(reviews))
    outer_dataset_id = run_sync(job(all_reviews))
    return outer_dataset_id


@app.task(name="extract-rollback_pipeline")
def rollback_pipeline(task_id, exc, traceback, dataset_id: UUID, provider_ref: UUID):
    async def rollback_job():
        uow = SessionFactory.create()
        async with uow as u:
            dataset_app = DatasetApplication(uow=u)
            source_app = DataSourceApplication(uow=u)
            status = {"status": DatasetStatus.failed}
            _edited_dataset = await dataset_app.edit_dataset(
                changed_value=status, id=dataset_id, commit=False
            )
            _is_source_removed = await source_app.remove_data_source(
                source_id=provider_ref
            )
            await u.commit()

    print(exc)
    print(traceback)
    run_sync(rollback_job())


@app.task(name="extract-extract_insight")
def extract_insight(dataset_id: UUID, language: Optional[str] = "indonesian"):
    extraction_uc = ExtractionUseCase(extraction_service=ExtractionService())

    async def fetch_reviews(d_id: UUID):
        uow = SessionFactory.create()
        async with uow as u:
            reviews = await extraction_uc.get_reviews_from_dataset(
                uow=u, dataset_id=d_id
            )
            return reviews

    outer_reviews = run_sync(fetch_reviews(dataset_id))
    positive_extract, negative_extract = extraction_uc.extract(
        reviews=outer_reviews,
        language=language,
    )
    return (dataset_id, positive_extract, negative_extract)


@app.task(name="extract-upsert_insights")
def upsert_insights(extracted_data):
    async def upsert_job(extracted: list[dict], dataset_id: UUID):
        insight_data = []
        mapped_review_ids = []
        for data in extracted:
            mapped_review_ids.append(data.pop("review_id"))
            insight_data.append(data)
        uow = SessionFactory.create()
        async with uow as u:
            insights_app = InsightApplication(uow=u)
            _is_upserted = await insights_app.upsert_insights(
                insights_data=insight_data,
                insight_review_mappings=mapped_review_ids,
                dataset_id=dataset_id,
            )
            if _is_upserted:
                await u.commit()

    dataset_id, pos_ext, neg_ext = extracted_data
    all_extracted = list(chain.from_iterable((pos_ext, neg_ext)))
    run_sync(
        upsert_job(extracted=all_extracted, dataset_id=dataset_id)
    )  # inject dataset id
    return (len(all_extracted), dataset_id)


@app.task(name="extract-finish_extract")
def finish_extract(payload):
    async def job(n_reviews: int, d_id):
        uow = SessionFactory.create()
        async with uow as u:
            dataset_app = DatasetApplication(uow=u)
            status = {
                "status": DatasetStatus.extracted,
                "is_empty": False,
                "total_reviews": n_reviews,
            }
            _edited_dataset = await dataset_app.edit_dataset(
                changed_value=status, id=d_id, commit=False
            )
            await u.commit()

    reviews_number, dataset_id = payload
    run_sync(job(n_reviews=reviews_number, d_id=dataset_id))
