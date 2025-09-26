cat > README.md << 'EOF'
# 🌟 Be-My-Eyes: Object Detection API

A powerful object detection system designed to assist visually impaired users with detailed image descriptions.

## 📌 Overview

**Be-My-Eyes** is an API that combines multiple AI technologies to provide detailed descriptions of images for visually impaired users:

- 🖼️ **YOLO Model**: Detects objects in images with confidence scores
- 🗣️ **Llama Vision (OpenRouter)**: Generates accessibility-focused captions
- 🔊 **gTTS**: Converts captions to speech for audio output

The system follows a clean **MVC architecture** with proper separation of concerns.

## ✨ Key Features

- 🗣️ **Accessibility-focused descriptions**: Natural language descriptions designed for visually impaired users
- 🔊 **Text-to-Speech**: Converts captions to audio files for easier consumption
- 📊 **Detailed object detection**: Identifies objects with positions (left/center/right) and confidence scores
- 🔒 **Secure file handling**: Validates file types and manages temporary storage
- 🧪 **Comprehensive error handling**: Clear error messages with detailed debugging information

## Installation

### Install the required packages

```bash
$ pip install -r requirements.txt
```

### Setup the environment variables

```bash
$ cp .env.example .env
```

Set your environment variables in the `.env` file. Like `OPENROUTER_API_KEY` value.

## Run Docker Compose Services

```bash
$ cd docker
$ cp .env.example .env
```

- update `.env` with your credentials



```bash
$ cd docker
$ sudo docker compose up -d
```

## Access Services

- **FastAPI**: http://localhost:8000
- **Flower Dashboard**: http://localhost:5555 (admin/password from env)
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090

## Run the FastAPI server (Development Mode)

```bash
$ uvicorn src.main:app --reload
```