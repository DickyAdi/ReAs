import pytest
import io
import pandas as pd
from fastapi.datastructures import UploadFile
from fastapi.responses import JSONResponse

from services import validation
from config.settings import MAX_SIZE_BYTES

@pytest.mark.asyncio
async def test_validation_valid():
    contents = b"review\nGreat place\nNice coffee"
    file = UploadFile(filename='test.csv', file=io.BytesIO(contents))
    response = await validation.validate_file(file, 'review')

    assert isinstance(response, dict)
    assert response['valid'] == True
    assert isinstance(response['data'], pd.DataFrame)
    assert 'review' in list(response['data'].columns)
    assert len(response['data']) == 2

@pytest.mark.asyncio
async def test_validation_not_csv_files():
    contents = b"review\nGreat place\nNice coffee"
    file = UploadFile(filename='test.txt', file=io.BytesIO(contents))
    response = await validation.validate_file(file, 'review')

    assert isinstance(response, dict)
    assert response['valid'] == False
    assert isinstance(response['response'], JSONResponse)
    assert response['response'].status_code == 415

@pytest.mark.asyncio
async def test_validation_oversize_file():
    contents = b"review\n" + b"A" * (MAX_SIZE_BYTES + 1)
    file = UploadFile(filename='test.csv', file=io.BytesIO(contents))
    response = await validation.validate_file(file, 'review')

    assert response['valid'] == False
    assert response['response'].status_code == 413

@pytest.mark.asyncio
async def test_validation_column_does_not_exists():
    contents = b"tests\nGreat coffee\ntesting"
    file = UploadFile(filename='test.csv', file=io.BytesIO(contents))
    response = await validation.validate_file(file, 'review')

    assert response['valid'] == False
    assert response['response'].status_code == 422