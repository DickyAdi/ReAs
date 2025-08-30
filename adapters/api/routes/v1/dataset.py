from fastapi import APIRouter, Depends, Body
from typing import Annotated

from ...services.dataset import get_dataset_app
from ...services.auth import min_tier, get_user_from_any_schema
from ...schemas.response import ResponseCreated
from infrastructure.db.models.user_model import User
from application.datasets import DatasetApplication
from domain.enums.tiers import Tier


router = APIRouter(
    prefix="/dataset",
    tags=["dataset"],
    dependencies=[Depends(min_tier(min_tier=Tier.Base))],
)


@router.post("/create")
async def create_dataset(
    dataset_name: Annotated[str, Body(embed=True)] = ...,
    dataset_app: DatasetApplication = Depends(get_dataset_app),
    curr_user: User = Depends(get_user_from_any_schema()),
):
    dataset = await dataset_app.create_dataset(
        dataset_name=dataset_name, issuer_id=curr_user.id
    )
    if dataset:
        return ResponseCreated(
            message="Dataset created",
            data={"dataset_name": dataset.name, "dataset_status": dataset.status.value},
        )
