from fastapi import APIRouter, Depends
from models.enums import ResponseEnums
from helpers import Settings, get_settings


health_router = APIRouter(
    prefix = "/api/v1",
    tags = ["api_v1"]
)

@health_router.get("/")
async def welcome(app_settings: Settings = Depends(get_settings)):

    app_name = app_settings.APP_NAME
    app_version = app_settings.APP_VERSION


    return {
        "message":      ResponseEnums.HEALTH_CHECK_GOOD.value,
        "app_name":     app_name,
        "app_version":  app_version,
        }
