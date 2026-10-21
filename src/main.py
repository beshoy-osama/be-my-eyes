import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from services import YOLOService, VLMService, TextToSpeechController
from helpers import get_settings
from routes import data_router, health_router, processes_router

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build application dependencies once before serving requests."""
    settings = get_settings()

    # 1. Initialize YOLO Controller
    yolo_client = YOLOService()
    yolo_client.set_model_path(settings.YOLO_MODEL_NAME)

    app.state.yolo_controller = yolo_client

    # 2. Initialize VLM Controller
    vlm_client = VLMService()
    await vlm_client.set_model(
        model_name=settings.VLM_MODEL_NAME,
        api_url=settings.VLM_API_URL,
        api_key=settings.OPENROUTER_API_KEY.get_secret_value(),
    )
    
    app.state.vlm_controller = vlm_client

    # # 3. Initialize TTS Controller
    # tts_controller = TextToSpeechController()
    # app.state.tts_controller = tts_controller

    logger.info("Application startup complete. Ready to serve requests!")

    yield

    await vlm_client.aclose()
    logger.info("Application shut down cleanly")


app = FastAPI(lifespan=lifespan)

app.include_router(health_router)
app.include_router(data_router)
app.include_router(processes_router)
