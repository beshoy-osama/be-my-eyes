import asyncio
import logging
import os
import time

from fastapi import APIRouter, Request, status
from fastapi.responses import FileResponse, JSONResponse

from services import DataController, YOLOService, VLMService, TextToSpeechService
from models.enums import ResponseEnums

from schemas import DetectionRequest

processes_router = APIRouter(
    prefix="/api/v1/processes",
    tags=["api_v1", "processes"],
)

logger = logging.getLogger(__name__)



# Stage 1: detection
@processes_router.post("/detection/{project_id}")
async def detection(request: Request, project_id: str, detection_request: DetectionRequest):
    start_time = time.perf_counter()

    yolo_client: YOLOService = request.app.state.yolo_controller

    image_path = DataController().get_file_path(file_id=detection_request.file_id, project_id=project_id)
    logger.info(f"Resolved image path: {image_path}")

    if not image_path:
        error_msg = f"File '{detection_request.file_id}' was not found in project '{project_id}'"
        logger.warning(error_msg)
        return JSONResponse(
            content={"signal": ResponseEnums.FILE_NOT_FOUND.value, "error": error_msg},
            status_code=status.HTTP_404_NOT_FOUND,
        )

    try:
        detection_result = await asyncio.to_thread(
            yolo_client.predict, image_path, detection_request.confidence
        )
    except FileNotFoundError as e:
        logger.warning(f"Image file not found during detection: {e}")
        return JSONResponse(
            content={"signal": ResponseEnums.FILE_NOT_FOUND.value, "error": str(e)},
            status_code=status.HTTP_404_NOT_FOUND,
        )
    except ValueError as e:
        logger.warning(f"Validation error during detection: {e}")
        return JSONResponse(
            content={"signal": ResponseEnums.DETECTION_FAILED.value, "error": str(e)},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        return JSONResponse(
            content={"signal": ResponseEnums.DETECTION_FAILED.value, "error": str(e)},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "signal": ResponseEnums.DETECTION_SUCCESS.value,
            "project_id": project_id,
            "file_id": detection_request.file_id,
            "detection": detection_result.model_dump(mode="json"),
            "processing_time": time.perf_counter() - start_time,
        },
    )


# Stage 2: detection + VLM
@processes_router.post("/detection-vlm/{project_id}")
async def detection_vlm(request: Request, project_id: str, detection_request: DetectionRequest):
    start_time = time.perf_counter()

    yolo_client: YOLOService = request.app.state.yolo_controller
    vlm_client: VLMService = request.app.state.vlm_controller

    image_path = DataController().get_file_path(file_id=detection_request.file_id, project_id=project_id)
    logger.info(f"Resolved image path: {image_path}")

    if not image_path:
        error_msg = f"File '{detection_request.file_id}' was not found in project '{project_id}'"
        logger.warning(error_msg)
        return JSONResponse(
            content={"signal": ResponseEnums.FILE_NOT_FOUND.value, "error": error_msg},
            status_code=status.HTTP_404_NOT_FOUND,
        )

    try:
        detection_result = await asyncio.to_thread(
            yolo_client.predict, image_path, detection_request.confidence
        )
    except FileNotFoundError as e:
        logger.warning(f"Image file not found during detection: {e}")
        return JSONResponse(
            content={"signal": ResponseEnums.FILE_NOT_FOUND.value, "error": str(e)},
            status_code=status.HTTP_404_NOT_FOUND,
        )
    except ValueError as e:
        logger.warning(f"Validation error during detection: {e}")
        return JSONResponse(
            content={"signal": ResponseEnums.DETECTION_FAILED.value, "error": str(e)},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        return JSONResponse(
            content={"signal": ResponseEnums.DETECTION_FAILED.value, "error": str(e)},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    try:
        caption_result = await vlm_client.generate_caption(
            image_path, detection_result.objects
        )
    except Exception as e:
        logger.error(f"Caption generation failed: {e}", exc_info=True)
        return JSONResponse(
            content={"signal": ResponseEnums.CAPTION_FAILED.value, "error": str(e)},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return JSONResponse(
    status_code=status.HTTP_200_OK,
    content={
        "signal": ResponseEnums.CAPTION_SUCCESS.value,
        "project_id": project_id,
        "file_id": detection_request.file_id,
        "detection": detection_result.model_dump(mode="json"),
        "caption": caption_result.model_dump(mode="json"),
        "processing_time": time.perf_counter() - start_time,
    },
)


# Stage 3: detection + VLM + TTS
@processes_router.post("/detection-vlm-tts/{project_id}")
async def detection_vlm_tts(request: Request, project_id: str, detection_request: DetectionRequest):
    start_time = time.perf_counter()

    yolo_client: YOLOService = request.app.state.yolo_controller
    vlm_client: VLMService = request.app.state.vlm_controller
    tts_client: TextToSpeechService = request.app.state.tts_controller

    image_path = DataController().get_file_path(file_id=detection_request.file_id, project_id=project_id)
    logger.info(f"Resolved image path: {image_path}")

    if not image_path:
        error_msg = f"File '{detection_request.file_id}' was not found in project '{project_id}'"
        logger.warning(error_msg)
        return JSONResponse(
            content={"signal": ResponseEnums.FILE_NOT_FOUND.value, "error": error_msg},
            status_code=status.HTTP_404_NOT_FOUND,
        )

    if detection_request.use_cache:
        cached_pipeline = tts_client.get_cached_pipeline(
            project_id=project_id,
            file_id=detection_request.file_id,
            confidence=detection_request.confidence,
            language=detection_request.language,
        )
        if cached_pipeline is not None:
            cached_speech = cached_pipeline["speech"]
            cached_speech.url = str(
                request.url_for(
                    "get_speech",
                    project_id=project_id,
                    file_name=cached_speech.file_name,
                )
            )
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "signal": ResponseEnums.SPEECH_SUCCESS.value,
                    "project_id": project_id,
                    "file_id": detection_request.file_id,
                    "detection": cached_pipeline["detection"].model_dump(mode="json"),
                    "caption": cached_pipeline["caption"].model_dump(mode="json"),
                    "speech": cached_speech.model_dump(mode="json"),
                    "processing_time": time.perf_counter() - start_time,
                },
            )

    try:
        detection_result = await asyncio.to_thread(
            yolo_client.predict, image_path, detection_request.confidence
        )
    except FileNotFoundError as e:
        logger.warning(f"Image file not found during detection: {e}")
        return JSONResponse(
            content={"signal": ResponseEnums.FILE_NOT_FOUND.value, "error": str(e)},
            status_code=status.HTTP_404_NOT_FOUND,
        )
    except ValueError as e:
        logger.warning(f"Validation error during detection: {e}")
        return JSONResponse(
            content={"signal": ResponseEnums.DETECTION_FAILED.value, "error": str(e)},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        return JSONResponse(
            content={"signal": ResponseEnums.DETECTION_FAILED.value, "error": str(e)},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    try:
        caption_result = await vlm_client.generate_caption(
            image_path, detection_result.objects
        )
    except Exception as e:
        logger.error(f"Caption generation failed: {e}", exc_info=True)
        return JSONResponse(
            content={"signal": ResponseEnums.CAPTION_FAILED.value, "error": str(e)},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    try:
        speech_result = await asyncio.to_thread(
            tts_client.generate_speech,
            caption_result.caption,
            project_id,
            detection_request.file_id,
            detection_request.language,
            False,
        )
    except Exception as e:
        logger.error(f"TTS generation failed: {e}", exc_info=True)
        return JSONResponse(
            content={"signal": ResponseEnums.SPEECH_FAILED.value, "error": str(e)},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    if detection_request.use_cache:
        tts_client.set_cached_pipeline(
            project_id=project_id,
            file_id=detection_request.file_id,
            confidence=detection_request.confidence,
            language=detection_request.language,
            detection=detection_result,
            caption=caption_result,
            speech=speech_result,
        )

    speech_result.url = str(
        request.url_for(
            "get_speech",
            project_id=project_id,
            file_name=speech_result.file_name,
        )
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "signal": ResponseEnums.SPEECH_SUCCESS.value,
            "project_id": project_id,
            "file_id": detection_request.file_id,
            "detection": detection_result.model_dump(mode="json"),
            "caption": caption_result.model_dump(mode="json"),
            "speech": speech_result.model_dump(mode="json"),
            "processing_time": time.perf_counter() - start_time,
        },
    )


# Stream generated speech file
@processes_router.get("/speech/{project_id}/{file_name}", name="get_speech", response_class=FileResponse)
async def get_speech(request: Request, project_id: str, file_name: str):
    tts_client: TextToSpeechService = request.app.state.tts_controller

    file_path = tts_client.get_speech_file_path(file_name=file_name, project_id=project_id)
    logger.info(f"Resolved speech file path: {file_path}")

    if file_path is None:
        error_msg = f"Speech file '{file_name}' was not found in project '{project_id}'"
        logger.warning(error_msg)
        return JSONResponse(
            content={
                "signal": ResponseEnums.FILE_NOT_FOUND.value,
                "error": error_msg,
            },
            status_code=status.HTTP_404_NOT_FOUND,
        )

    logger.info(f"Streaming speech file: '{file_path}'")
    return FileResponse(
        path=file_path,
        media_type="audio/mpeg",
        filename=os.path.basename(file_path),
        content_disposition_type="inline",
    )