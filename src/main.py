import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from controllers import TextToSpeechController, VLMController, YOLOController
from helpers import get_settings
from routes import data_router, health_router, processes_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build application dependencies once before serving requests."""
    settings = get_settings()

    # 1. Initialize YOLO Controller and set model path
    yolo_controller = YOLOController()
    yolo_controller.set_model_path(settings.YOLO_MODEL_NAME)
    yolo_controller.get_model()
    app.state.yolo_controller = yolo_controller

    # # 2. Initialize VLM Controller
    # vlm_controller = VLMController()
    # app.state.vlm_controller = vlm_controller

    # # 3. Initialize TTS Controller
    # tts_controller = TextToSpeechController()
    # app.state.tts_controller = tts_controller

    print(f"[{time.strftime('%X')}] Application startup complete. Ready to serve requests!")

    yield

    print(f"[{time.strftime('%X')}] Application shut down cleanly")


app = FastAPI(lifespan=lifespan)

app.include_router(health_router)
app.include_router(data_router)
app.include_router(processes_router)
