import os
import sys

# Add project root to sys.path to allow importing 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# CRITICAL: Apply compatibility patches BEFORE any speechbrain imports
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    def _list_audio_backends():
        return ["soundfile"]
    torchaudio.list_audio_backends = _list_audio_backends

import huggingface_hub
_original_hf_hub_download = huggingface_hub.hf_hub_download
def _patched_hf_hub_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        token_val = kwargs.pop("use_auth_token")
        if "token" not in kwargs:
            kwargs["token"] = token_val
    return _original_hf_hub_download(*args, **kwargs)
huggingface_hub.hf_hub_download = _patched_hf_hub_download

import time
import base64
import traceback
from fastapi import FastAPI, HTTPException, Header, Body
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from dotenv import load_dotenv

# Import the new pipeline
try:
    import src
    print(f"DEBUG: src module found at {src.__file__}")
    from src.pipeline.detector import VoicePipeline
except ImportError as e:
    # Fallback or detailed error logging
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f"Failed to import src.pipeline.detector.")
    print(f"CWD: {os.getcwd()}")
    print(f"Path: {sys.path}")
    print(f"Root dir: {root_path}")
    if os.path.exists(root_path):
        print(f"Contents of root: {os.listdir(root_path)}")
        src_path = os.path.join(root_path, "src")
        if os.path.exists(src_path):
             print(f"Contents of src: {os.listdir(src_path)}")
    raise e

load_dotenv()

app = FastAPI(title="Voice Detector API (Refactored)")

# Initialize Pipeline (Single instance)
# Config path relative to execution root or use absolute
pipeline = VoicePipeline("config/hparams.yaml")

API_KEY = os.getenv("API_KEY", "your-secret-api-key")

class VoiceDetectionRequest(BaseModel):
    language: str = "en"
    audioFormat: str = "mp3"
    audioBase64: str

@app.on_event("startup")
async def startup_event():
    # Warmup if needed
    pass

@app.post("/api/voice-detection")
async def detect_voice(
    x_api_key: str = Header(None),
    request_data: VoiceDetectionRequest = Body(...)
):
    # 1. API Key Validation
    # Allow fallback key for testing if needed
    expected_key = os.getenv("API_KEY", "test_key_123")
    if x_api_key and x_api_key != expected_key and x_api_key != API_KEY:
         raise HTTPException(status_code=403, detail="Invalid API key")

    start_time = time.time()

    try:
        # 2. Decode Audio
        try:
            audio_bytes = base64.b64decode(request_data.audioBase64, validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid Base64 string")
            
        # 3. Process via Pipeline
        result = pipeline.process(audio_bytes)
        
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])
        
        # 4. Construct Response
        response_payload = {
            "status": "success",
            "language": request_data.language,
            "classification": result["classification"],
            "confidenceScore": result["confidenceScore"],
            "explanation": result["explanation"],
            "processingTime": f"{time.time() - start_time:.2f}s",
            "details": result.get("details", {})
        }
        
        return JSONResponse(content=response_payload)

    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "VoiceGuard API Running (Refactored Structure)"}
