from fastapi import (
    APIRouter,
    status,
    UploadFile,
    File,
    Request,
    Depends,
    Form,
)
from fastapi.responses import JSONResponse
import asyncio

from application.use_cases.pipeline import PipelineExtractCsvFlow

from infrastructure.db.models import User
from domain.enums.tiers import Tier

from ...context.limiter import limiter
from ...core.validation.csv import validate_csv_metadata
from ...services.auth import min_tier
from ...services.extraction import get_extract_csv_flow

router = APIRouter(tags=["feature"])


@router.post("/extract")
@limiter.limit("2/second;10/minute;30/day")
async def extract(
    request: Request,
    text_column: str = Form(...),
    file: UploadFile = File(...),
    flow: PipelineExtractCsvFlow = Depends(get_extract_csv_flow),
):
    validated_csv_content = await validate_csv_metadata(file)
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(
        request.app.state.executor,
        flow,
        validated_csv_content,
        text_column,
        request.app.state.model,
    )
    response_body = JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": "Extraction successful.",
            "data": {
                "positive": {
                    "trend_topics": res["positive_trend_topics"].to_dict(
                        orient="records"
                    ),
                    "frequent_topics": res["positive_frequent_topics"].to_dict(
                        orient="records"
                    ),
                    "count": res["n_positive"],
                },
                "negative": {
                    "trend_topics": res["negative_trend_topics"].to_dict(
                        orient="records"
                    ),
                    "frequent_topics": res["negative_frequent_topics"].to_dict(
                        orient="records"
                    ),
                    "count": res["n_negative"],
                },
                "number_valid_rows": int(res["total_valid_reviews"]),
            },
        },
    )
    return response_body


@router.get("/extract_map")
async def extract_map(user: User = Depends(min_tier(min_tier=Tier.Mid))):
    return {"message": "Hey im premium feature"}
