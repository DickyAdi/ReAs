from fastapi import UploadFile, status
from fastapi.responses import JSONResponse
import io
import os
import pandas as pd
from config.settings import settings

async def validate_file(file: UploadFile, text_column:str):
    _, ext = os.path.splitext(file.filename)
    if ext != '.csv':
        return {
            'valid' : False,
            'response' : JSONResponse(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                content={
                    'status' : 'error',
                    'code' : 415,
                    'message' : 'Unsupported file type. Only .csv files are allowed.'
                }
            )
        }
    try:
        contents = await file.read()
        if len(contents) > settings.max_size_bytes:
            return {
                'valid' : False,
                'response' : JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        'status' : 'error',
                        'code' : 413,
                        'message' : f'File size exceeds {settings.max_size_mb} MB limit.'
                    }
                )
            }
        df = pd.read_csv(io.BytesIO(contents))
        if text_column not in list(df.columns):
            return {
                'valid' : False,
                'response' : JSONResponse(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    content={
                        'status' : 'error',
                        'code' : 422,
                        'message' : f'Column does not exists. {text_column} Not in {list(df.columns)}'
                    }
                )
            }
        return {
            'valid' : True,
            'data' : df
        }
    except Exception as e:
        return {
            'valid' : False,
            'response' : JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    'status' : 'error',
                    'code' : 422,
                    'message' : f'Could not read csv. {str(e)}'
                }
            )
        }