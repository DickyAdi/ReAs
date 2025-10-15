from fastapi import APIRouter, Depends, Body, Form, File, UploadFile
from typing import Annotated, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from ...services.auth import min_tier, get_user_from_any_schema
from ...services.rate_limiter import is_allowed
from ...core.validation.csv import validate_csv_metadata
from ...core.validation.gmaps_url import validate_gmaps_url
from ...schemas.response import ResponseCreated, ResponseAccepted, ResponseOk
from infrastructure.db.models.user_model import User
from infrastructure.db import UnitOfWork, get_db
from infrastructure.workers import DatasetPipeline
from application.datasets import DatasetApplication
from application.use_cases.pipeline import DatasetPipelineOrchestrator
from application.use_cases.insight import InsightsUseCase
from domain.enums.tiers import Tier
from domain.enums.texts import TextLanguage
from domain.enums.datasets import DatasetProvider


router = APIRouter(
    prefix="/dataset",
    tags=["dataset"],
    dependencies=[Depends(min_tier(min_tier=Tier.Base))],
)


@router.post("/create")
async def create_dataset(
    dataset_name: Annotated[str, Body(embed=True)] = ...,
    db: AsyncSession = Depends(get_db),
    curr_user: User = Depends(get_user_from_any_schema()),
):
    uow = UnitOfWork(db=db)
    async with uow as u:
        dataset_app = DatasetApplication(uow=u)
        dataset = await dataset_app.create_dataset(
            dataset_name=dataset_name, issuer_id=curr_user.id
        )
    if dataset:
        return ResponseCreated(
            message="Dataset created",
            data={
                "dataset_name": dataset.name,
                "dataset_status": dataset.status.value,
                "dataset_id": dataset.id.hex,
            },
        )


@router.post("/extract")
async def upload_csv(
    text_column: str = Form(...),
    file: UploadFile = File(...),
    dataset_id: str = Form(...),
    rating_column: Optional[str] = Form(),
    db: AsyncSession = Depends(get_db),
    _is_rate_limited=Depends(is_allowed),
):
    dataset_id = UUID(hex=dataset_id)
    secured_file = await validate_csv_metadata(file=file)
    worker = DatasetPipelineOrchestrator(
        service=DatasetPipeline(
            data=secured_file, text_column=text_column, rating_column=rating_column
        ),
        uow=UnitOfWork(db=db),
    )
    task_id = await worker.run(
        source=DatasetProvider.local, language=TextLanguage.ID, dataset_id=dataset_id
    )
    return ResponseAccepted(
        message="Data received. Processing", data={"task_id": task_id}
    )


@router.post("/extract/scrape")
async def extract_scrape(
    gmaps_url: str = Form(...),
    dataset_id: str = Form(...),
    db: AsyncSession = Depends(get_db),
    _is_rate_limited=Depends(is_allowed),
):
    dataset_id = UUID(hex=dataset_id)
    secured_url = validate_gmaps_url(gmaps_url)
    worker = DatasetPipelineOrchestrator(
        uow=UnitOfWork(db=db),
        service=DatasetPipeline(
            data=secured_url, text_column="None", rating_column="None"
        ),
    )
    task_id = await worker.run(
        source=DatasetProvider.scrape, language=TextLanguage.ID, dataset_id=dataset_id
    )
    return ResponseAccepted(
        message="Data received. Processing", data={"task_id": task_id}
    )


@router.get("/insights/{dataset_id}")
async def get_insights(
    dataset_id: str,
    offset: Optional[int],
    limit: Optional[int],
    db: AsyncSession = Depends(get_db),
):
    real_uuid = UUID(hex=dataset_id)
    uow = UnitOfWork(db=db)
    uc = InsightsUseCase(uow=uow)
    data = await uc.get_review_insight(dataset_id=real_uuid, offset=offset, limit=limit)
    return ResponseOk(message="Ok", data=data)
