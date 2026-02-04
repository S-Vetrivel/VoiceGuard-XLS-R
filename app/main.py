from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Body, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import Optional
from app.audio import process_audio
from app.infer import VoiceClassifier
from dotenv import load_dotenv
import os
import traceback

load_dotenv()

app = FastAPI(title="Voice Detector API")

# Singleton Classifier
classifier = None

def get_classifier():
    global classifier
    if classifier is None:
        classifier = VoiceClassifier()
    return classifier

API_KEY = os.getenv("API_KEY", "your-secret-api-key")

# Pydantic Model for Strict Request Body
class VoiceDetectionRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.on_event("startup")
async def startup_event():
    get_classifier()

# Custom Exception Handler for strict error format
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
     return JSONResponse(
        status_code=400,
        content={"status": "error", "message": "Invalid API key or malformed request"},
    )


@app.post("/api/voice-detection")
async def detect_voice(
    x_api_key: Optional[str] = Header(None),
    request_data: VoiceDetectionRequest = Body(...)
):
    # 1. API Key Validation
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key or malformed request")

    # 2. Format Validation
    if request_data.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only 'mp3' format is supported")
        
    try:
        classifier_instance = get_classifier()
        
        # 3. Process Audio (decodes Base64 -> WAV -> 16kHz Mono)
        waveform = process_audio(request_data.audioBase64)
        
        if waveform is None:
             raise HTTPException(status_code=400, detail="Could not process audio.")
             
        # 4. Predict
        result = classifier_instance.predict(waveform, language=request_data.language)
        
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])
             
        # 5. Construct Strict JSON Response
        response_payload = {
            "status": "success",
            "language": request_data.language,
            "classification": result["prediction"], # "AI_GENERATED" or "HUMAN"
            "confidenceScore": result["confidence"],
            "explanation": result["explanation"]
        }
        
        return JSONResponse(content=response_payload)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Audio processing error: {str(ve)}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/")
async def root():
    return {"message": "Voice Detector API is running. POST /api/voice-detection"}
