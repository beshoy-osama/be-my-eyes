import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.responses import FileResponse, JSONResponse

from controllers import DataController, TextToSpeechController, VLMController, YOLOController
from models.enums import ResponseSignal
from routes.schemes import ProcessRequest
from schemas import (
    DetectionCaptionResponse,
    DetectionCaptionSpeechResponse,
    DetectionResponse,
)

processes_router = APIRouter(
    prefix="/api/v1/processes",
    tags=["api_v1", "processes"],
)

logger = logging.getLogger(__name__)


# ---------------- Helper Functions ---------------- #

def _resolve_image_path(project_id: str, file_id: str) -> tuple[str | None, JSONResponse | None]:
    """Resolve and validate the uploaded file path."""
    image_path = DataController().get_file_path(file_id=file_id, project_id=project_id)
    if image_path is None:
        return None, JSONResponse(
            content={
                "signal": ResponseSignal.FILE_NOT_FOUND.value,
                "error": f"File '{file_id}' was not found in project '{project_id}'",
            },
            status_code=status.HTTP_404_NOT_FOUND,
        )
    return image_path, None


async def _run_detection(
    yolo_controller: YOLOController,
    image_path: str,
    confidence: float | None,
) -> tuple[Any | None, JSONResponse | None]:
    """Execute YOLO prediction with handled exceptions."""
    try:
        detection_result = await asyncio.to_thread(
            yolo_controller.predict, image_path, confidence
        )
        return detection_result, None
    except ValueError as e:
        return None, JSONResponse(
            content={"signal": ResponseSignal.INVALID_CONFIDENCE.value, "error": str(e)},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        return None, JSONResponse(
            content={"signal": ResponseSignal.DETECTION_FAILED.value, "error": str(e)},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


async def _run_caption(
    vlm_controller: VLMController,
    image_path: str,
    objects: Any,
) -> tuple[Any | None, JSONResponse | None]:
    """Execute VLM captioning with handled exceptions."""
    try:
        caption_result = await vlm_controller.generate_caption(image_path, objects)
        return caption_result, None
    except Exception as e:
        logger.error(f"VLM failed: {e}", exc_info=True)
        return None, JSONResponse(
            content={"signal": ResponseSignal.CAPTION_FAILED.value, "error": str(e)},
            status_code=status.HTTP_502_BAD_GATEWAY,
        )


# ---------------- Endpoints ---------------- #

# Stage 1: detection
@processes_router.post("/detection/{project_id}", response_model=DetectionResponse)
async def detection(request: Request, project_id: str, process_request: ProcessRequest):
    start_time = time.perf_counter()

    yolo_controller: YOLOController = request.app.state.yolo_controller

    # Resolve image path
    image_path, error_response = _resolve_image_path(project_id, process_request.file_id)
    if error_response:
        return error_response

    # Run detection
    detection_result, error_response = await _run_detection(
        yolo_controller, image_path, process_request.confidence
    )
    if error_response:
        return error_response

    return DetectionResponse(
        signal=ResponseSignal.DETECTION_SUCCESS.value,
        project_id=project_id,
        file_id=process_request.file_id,
        detection=detection_result,
        processing_time=time.perf_counter() - start_time,
    )


# # Stage 2: detection + VLM
# @processes_router.post("/detection-vlm/{project_id}", response_model=DetectionCaptionResponse)
# async def detection_vlm(request: Request, project_id: str, process_request: ProcessRequest):
#     start_time = time.perf_counter()
#     yolo_controller: YOLOController = request.app.state.yolo_controller
#     vlm_controller: VLMController = request.app.state.vlm_controller

#     # Resolve image path
#     image_path, error_response = _resolve_image_path(project_id, process_request.file_id)
#     if error_response:
#         return error_response

#     # Run detection
#     detection_result, error_response = await _run_detection(
#         yolo_controller, image_path, process_request.confidence
#     )
#     if error_response:
#         return error_response

#     # Run VLM caption
#     caption_result, error_response = await _run_caption(
#         vlm_controller, image_path, detection_result.objects
#     )
#     if error_response:
#         return error_response

#     return DetectionCaptionResponse(
#         signal=ResponseSignal.CAPTION_SUCCESS.value,
#         project_id=project_id,
#         file_id=process_request.file_id,
#         detection=detection_result,
#         caption=caption_result,
#         processing_time=time.perf_counter() - start_time,
#     )


# # Stage 3: detection + VLM + TTS
# @processes_router.post(
#     "/detection-vlm-tts/{project_id}", response_model=DetectionCaptionSpeechResponse
# )
# async def detection_vlm_tts(request: Request, project_id: str, process_request: ProcessRequest):
#     start_time = time.perf_counter()
#     yolo_controller: YOLOController = request.app.state.yolo_controller
#     vlm_controller: VLMController = request.app.state.vlm_controller
#     tts_controller: TextToSpeechController = request.app.state.tts_controller

#     # Resolve image path
#     image_path, error_response = _resolve_image_path(project_id, process_request.file_id)
#     if error_response:
#         return error_response

#     # Run detection
#     detection_result, error_response = await _run_detection(
#         yolo_controller, image_path, process_request.confidence
#     )
#     if error_response:
#         return error_response

#     # Run VLM caption
#     caption_result, error_response = await _run_caption(
#         vlm_controller, image_path, detection_result.objects
#     )
#     if error_response:
#         return error_response

#     # Run Text-to-speech
#     try:
#         speech_result = await asyncio.to_thread(
#             tts_controller.generate_speech, caption_result.caption, process_request.language
#         )
#     except Exception as e:
#         logger.error(f"TTS failed: {e}", exc_info=True)
#         return JSONResponse(
#             content={"signal": ResponseSignal.SPEECH_FAILED.value, "error": str(e)},
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#         )

#     speech_result.url = str(request.url_for("get_speech", file_name=speech_result.file_name))

#     return DetectionCaptionSpeechResponse(
#         signal=ResponseSignal.SPEECH_SUCCESS.value,
#         project_id=project_id,
#         file_id=process_request.file_id,
#         detection=detection_result,
#         caption=caption_result,
#         speech=speech_result,
#         processing_time=time.perf_counter() - start_time,
#     )


# # Stream generated speech file
# @processes_router.get("/speech/{file_name}", name="get_speech")
# async def get_speech(request: Request, file_name: str):
#     tts_controller: TextToSpeechController = request.app.state.tts_controller

#     file_path = tts_controller.get_speech_file_path(file_name)
#     if file_path is None:
#         return JSONResponse(
#             content={
#                 "signal": ResponseSignal.FILE_NOT_FOUND.value,
#                 "error": f"Speech file '{file_name}' was not found",
#             },
#             status_code=status.HTTP_404_NOT_FOUND,
#         )

#     return FileResponse(
#         path=file_path,
#         media_type="audio/mpeg",
#         filename=file_name,
#         content_disposition_type="inline",
#     )