# 👀 Be-My-Eyes: Assistive Vision & Scene Description API

**Be-My-Eyes** is an end-to-end assistive AI backend built with **FastAPI** to help visually impaired users understand their surroundings. It combines real-time object detection (**YOLO**), contextual scene description via Vision-Language Models (**OpenRouter VLM**), and natural speech synthesis (**gTTS**) into a multi-stage processing pipeline.

---

## 📌 Overview & Processing Pipeline

The system processes uploaded images across three progressive stages:

1. **Stage 1 — Object Detection (`YOLOService`)**:
   - Detects objects in the uploaded image using **Ultralytics YOLO** (`yolo12s.pt`).
   - Computes each object's horizontal relative position (`left`, `center`, `right`) and confidence score.
2. **Stage 2 — Scene Captioning (`VLMService`)**:
   - Sends the encoded image alongside the spatial object detections to a Vision-Language Model via **OpenRouter**.
   - Uses structured **System** and **User** prompts tailored for visually impaired users to generate a concise, natural-language paragraph describing prominent elements, spatial layout, and scene context.
3. **Stage 3 — Speech Synthesis & Caching (`TextToSpeechService`)**:
   - Converts the generated caption into an `.mp3` audio file using **Google Text-to-Speech (`gTTS`)**.
   - Organizes generated audio per project (`assets/tts_output/<project_id>/<image_name>.mp3`) and provides a direct streaming endpoint.
   - Includes a **Full Pipeline Cache** to serve repeated requests for the same image instantaneously without re-running inference.

---

## ✨ Key Features

- 🗣️ **Accessibility-First Prompting**: Dedicated system and user prompts designed specifically for visually impaired users (clear spatial context, under five sentences, no markdown clutter).
- ⚡ **Asynchronous & Non-Blocking Architecture**:
  - Async file streaming with `aiofiles`.
  - Non-blocking HTTP connection pooling with `httpx.AsyncClient`.
  - CPU/GPU-bound YOLO inference and TTS synthesis offloaded via `asyncio.to_thread` with thread-safe locking.
- 🗂️ **Project-Scoped Storage**: Uploaded images and generated `.mp3` audio files are isolated by `project_id`.
- 🚀 **Full Pipeline Caching**: Caches `(detection, caption, speech)` results per `(project_id, file_id, confidence, language)` with an optional `use_cache` toggle.
- 🔒 **Strict File Validation**: Validates file signatures (magic bytes) via `filetype` and enforces maximum upload size limits.

---

## 📂 Project Structure

```text
be-my-eyes/
├── README.md
└── src/
    ├── main.py                      # FastAPI application entrypoint & lifespan manager
    ├── requirements.txt             # Python dependencies
    ├── .env                         # Environment variables configuration
    ├── .env.example                 # Example environment variables template
    │
    ├── assets/                      # Local storage for models and project files
    │   ├── files/                   # Uploaded images organized by <project_id>/
    │   ├── models/                  # YOLO weights (e.g., yolo12s.pt)
    │   └── tts_output/              # Generated MP3 audio organized by <project_id>/
    │
    ├── helpers/                     # Configuration & environment settings
    │   ├── __init__.py
    │   └── configuration.py         # Pydantic BaseSettings (Settings, get_settings)
    │
    ├── models/                      # Domain constants, enums, and prompt templates
    │   ├── prompts.py               # VLM system and user prompt builders
    │   └── enums/
    │       ├── __init__.py
    │       ├── PositionEnums.py     # Object spatial positions (left, center, right)
    │       └── ResponseEnums.py     # Standardized API response signals
    │
    ├── routes/                      # FastAPI routers & endpoint handlers
    │   ├── __init__.py
    │   ├── health.py                # GET  /api/v1/ health check endpoint
    │   ├── data.py                  # POST /api/v1/data/upload/{project_id}
    │   └── processes.py             # Processing endpoints (YOLO, VLM, TTS, audio stream)
    │
    ├── schemas/                     # Pydantic request & response models
    │   ├── __init__.py
    │   ├── YOLOSchemas.py           # DetectionRequest, DetectedObject, DetectionResult
    │   ├── VLMSchemas.py            # CaptionResponse
    │   └── TTSSchemas.py           # SpeechResult
    │
    └── services/                    # Core business logic & AI service controllers
        ├── __init__.py
        ├── BaseController.py        # Shared base paths, settings, and utilities
        ├── DataController.py        # File validation, sanitization, and path resolution
        ├── YOLOService.py           # Thread-safe YOLO model loading & object detection
        ├── VLMService.py            # Async OpenRouter VLM client & caption generation
        └── TextToSpeechService.py   # gTTS synthesis, project audio storage & pipeline cache
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/beshoy-osama/be-my-eyes.git
cd be-my-eyes/src
```

### 2. Create a Virtual Environment & Install Dependencies

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the example environment file inside the `src/` directory:

```bash
cp .env.example .env
```

Then update your configuration values in `.env` (such as `OPENROUTER_API_KEY` and `YOLO_MODEL_NAME`).

### 4. Run the Server

Run Uvicorn from inside the `src/` directory:

```bash
uvicorn main:app --reload
```

Interactive API documentation will be available at:
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

---

## 📡 API Endpoints

### 1. Health Check
- **`GET /api/v1/`**
  - Returns application name, version, and health status signal.

### 2. Upload Image
- **`POST /api/v1/data/upload/{project_id}`**
  - **Form Data**: `file` (`.png`, `.jpg`, `.jpeg`)
  - Validates file signature and size, stores the image in `src/assets/files/{project_id}/`, and returns `file_id`.

### 3. Stage 1 — Object Detection Only
- **`POST /api/v1/processes/detection/{project_id}`**
  - **Request Body**:
    ```json
    {
      "file_id": "nzk7aqt7jf5z_cat.jpg",
      "confidence": 0.5
    }
    ```
  - **Response**: Detected objects with `object` class name, `position` (`left`, `center`, `right`), `confidence`, image dimensions, and processing time.

### 4. Stage 2 — Object Detection + VLM Scene Caption
- **`POST /api/v1/processes/detection-vlm/{project_id}`**
  - **Request Body**:
    ```json
    {
      "file_id": "nzk7aqt7jf5z_cat.jpg",
      "confidence": 0.5
    }
    ```
  - **Response**: YOLO detection results + natural-language `caption` generated by the VLM.

### 5. Stage 3 — Full Pipeline (Detection + VLM + TTS)
- **`POST /api/v1/processes/detection-vlm-tts/{project_id}`**
  - **Request Body**:
    ```json
    {
      "file_id": "nzk7aqt7jf5z_cat.jpg",
      "confidence": 0.5,
      "language": "en",
      "use_cache": true
    }
    ```
  - **Response**:
    ```json
    {
      "signal": "SPEECH_SUCCESS",
      "project_id": "2",
      "file_id": "nzk7aqt7jf5z_cat.jpg",
      "detection": { ... },
      "caption": {
        "caption": "A tabby cat is resting in the center of the frame...",
        "model_name": "google/gemma-3-27b-it:free",
        "processing_time": 2.84
      },
      "speech": {
        "file_name": "nzk7aqt7jf5z_cat.mp3",
        "file_path": ".../src/assets/tts_output/2/nzk7aqt7jf5z_cat.mp3",
        "file_size": 48192,
        "language": "en",
        "cached": false,
        "processing_time": 0.72,
        "url": "http://127.0.0.1:8000/api/v1/processes/speech/2/nzk7aqt7jf5z_cat.mp3"
      },
      "processing_time": 9.12
    }
    ```

### 6. Stream / Download Generated Speech Audio
- **`GET /api/v1/processes/speech/{project_id}/{file_name}`**
  - Streams the generated `.mp3` audio file (`audio/mpeg`) for the specified `project_id` and `file_name`.
