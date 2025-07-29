from fastapi import UploadFile, HTTPException, status


from config.settings import settings

async def validate_csv_metadata(file:UploadFile):
    try:
        if file.content_type not in ['text/csv', 'application/vnd.ms-excel']:
            return HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f'Unsupported file type. Only csv files allowed.'
            )
        contents = await file.read()
        if len(contents) > settings.max_size_bytes:
            return HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, 
                                detail=f'File size exceeds {settings.max_size_mb} MB limit.')
        file.file.seek(0)
        return contents
    except Exception:
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected Exception"
        )