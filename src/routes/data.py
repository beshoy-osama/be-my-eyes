from fastapi import APIRouter, Depends, UploadFile, status
from fastapi.responses import JSONResponse

from controllers import DataController
from models.enums import ResponseSignal
from helpers import Settings, get_settings


import logging
import aiofiles

data_router = APIRouter(
        prefix= "/api/v1/data",
        tags=['api_v1', 'data']
        )

logger = logging.getLogger(__name__)

# Upload file endpoint
@data_router.post('/upload/{project_id}')
async def upload_file(project_id: str, file: UploadFile,
                        app_settings: Settings = Depends(get_settings)
                        ):

        # validate file properties
        is_valid, signal_result = await DataController().validate_upload_file(file=file)

        if not is_valid:
                return JSONResponse(
                        content={
                                'signal': signal_result
                                },
                        status_code=status.HTTP_400_BAD_REQUEST)

        if not file.filename:
                return JSONResponse(
                        content={
                                'signal': ResponseSignal.FILE_UPLOAD_FAILED.value
                                },
                        status_code=status.HTTP_400_BAD_REQUEST)

        file_path, file_id = DataController().generate_unique_file_path(file_name=file.filename, project_id=project_id)

        try:
                async with aiofiles.open(file_path, "wb") as f:
                        while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                                await f.write(chunk)
        except Exception as e:

                logger.error(f'error while uploading file: {e}')

                return JSONResponse(
                                content={
                                        'response signal': ResponseSignal.FILE_UPLOAD_FAILED.value
                                                },
                                                status_code=status.HTTP_400_BAD_REQUEST
                                                )


        return JSONResponse(
                content={
                        'response signal': ResponseSignal.FILE_UPLOAD_SUCCESS.value,
                        'file_id': file_id,
                        'file_path': file_path
                                }
                                )
