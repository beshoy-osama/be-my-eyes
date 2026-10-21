import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.responses import FileResponse, JSONResponse

from controllers import DataController, TextToSpeechController, VLMController, YOLOController
from models.enums import ResponseEnums
from routes.schemes import DetectionSchemas, DetectionRequest
# from schemas import (
#     DetectionCaptionResponse,
#     DetectionCaptionSpeechResponse,
#     DetectionResponse,
# )

processes_router = APIRouter(
    prefix="/api/v1/processes",
    tags=["api_v1", "processes"],
)

logger = logging.getLogger(__name__)



# Stage 1: detection
@processes_router.post("/detection/{project_id}")
async def detection(request: Request, project_id: str, detection_request: DetectionRequest):
    start_time = time.perf_counter()

    yolo_controller: YOLOController = request.app.state.yolo_controller

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
            yolo_controller.predict, image_path, detection_request.confidence
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
#         signal=ResponseEnums.CAPTION_SUCCESS.value,
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
#             content={"signal": ResponseEnums.SPEECH_FAILED.value, "error": str(e)},
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#         )

#     speech_result.url = str(request.url_for("get_speech", file_name=speech_result.file_name))

#     return DetectionCaptionSpeechResponse(
#         signal=ResponseEnums.SPEECH_SUCCESS.value,
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
#                 "signal": ResponseEnums.FILE_NOT_FOUND.value,
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