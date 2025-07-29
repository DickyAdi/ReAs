from fastapi import APIRouter, status, UploadFile, File, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
import asyncio

from application.parse_csv_service import ParseCsvService
from infrastructure.io.parse_csv import CsvParser
from infrastructure.pipeline.extraction import ExtractionPipeline
from infrastructure.db.models import User
from domain.tier import Tier

from ...context.limiter import limiter
from ...core.validation.csv import validate_csv_metadata
from ...services.auth import get_current_active_user

router = APIRouter(tags=['feature'])

@router.post('/extract')
@limiter.limit("2/second;10/minute;30/day")
async def extract(request: Request, text_column:str, file: UploadFile = File(...)):
    contents = await validate_csv_metadata(file)
    if isinstance(contents, HTTPException): #return early due to invalid metadata
        return contents
    parser = ParseCsvService(contents, CsvParser())
    df = parser.run()
    pipe = ExtractionPipeline()
    try:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(request.app.executor, pipe.extract, df, text_column)
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong."
        )
    except:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Something went wrong.'
        )
    # res = pipe.extract(df, text_column)
    response_body = JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            'message' : 'Extraction successful.',
            'data' : {
                'positive' : {
                    'trend_topics' : res['positive_trend_topics'].to_dict(orient='records'),
                    'frequent_topics' : res['positive_frequent_topics'].to_dict(orient='records'),
                    'count' : res['n_positive']
                },
                'negative' : {
                    'trend_topics' : res['negative_trend_topics'].to_dict(orient="records"),
                    'frequent_topics' : res['negative_frequent_topics'].to_dict(orient='records'),
                    'count' : res['n_negative']
                },
                'number_valid_rows' : int(res['len_valid_mask'])
            }
        }
    )
    return response_body

@router.get('/extract_map')
async def extract_map(user:User=Depends(get_current_active_user)):
    if user.tier not in [Tier.Mid, Tier.Pro]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only Mid and Pro tier allowed."
        )
    return {'message' : 'Hey im premium feature'}