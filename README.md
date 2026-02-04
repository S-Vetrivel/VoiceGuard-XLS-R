---
title: VoiceGuard API
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# AI-Generated Voice Detector API

A production-ready REST API that accurately detects whether a given voice recording is **AI-generated** or **Human**.  
Built for the **AI-Generated Voice Detection Challenge** with specific support for **Tamil, English, Hindi, Malayalam, and Telugu**.

---

## 🚀 Features

- **Multilingual Support**: Uses the **XLS-R (Cross-Lingual Speech Representation)** model (`wav2vec2-large-xlsr-53`) pre-trained on 53 languages.
- **Strict API Specification**: Compliant with challenge requirements (Base64 MP3 input, standardized JSON response).
- **Hybrid Detection**: Combines Deep Learning embeddings with **Acoustic Feature Analysis** (Pitch Variance) for robust detection.
- **Explainability**: Provides human-readable explanations for every decision.
- **Secure**: Protected via `x-api-key` header authentication.

---

## 🛠️ Tech Stack

- **Framework**: FastAPI (Python)
- **Model**: PyTorch + HuggingFace Transformers (`facebook/wav2vec2-large-xlsr-53`)
- **Audio Processing**: `pydub` (ffmpeg) + `librosa`
- **Deployment**: Uvicorn

---

## 📥 Installation

### 1. Pre-requisites
- **Python 3.8+**
- **FFmpeg**: Required for audio processing (`pydub`).
  - **Linux**: `sudo apt install ffmpeg`
  - **Windows**: [Download here](https://ffmpeg.org/download.html) and add to Path.

### 2. Setup (Linux / macOS)
```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup (Windows)
```powershell
# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Environment
Create a `.env` file in the root directory:
```bash
API_KEY=test-key-123
```

---

## ▶️ Running the Server

**Universal Command:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
*The server will start at `http://localhost:8000`.*

---

## 📡 API Usage

### Endpoint: `POST /api/voice-detection`

#### Headers
| Key | Value |
| -- | -- |
| `x-api-key` | `your-secret-key-123` |
| `Content-Type` | `application/json` |

#### Request Body
```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "<BASE64_ENCODED_MP3_STRING>"
}
```

#### Response Example
```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "HUMAN",
  "confidenceScore": 0.98,
  "explanation": "High pitch variance and natural prosody detected."
}
```

---

## 🧪 Testing

### 1. Run the Verification Script
We have a built-in test suite that verifies the audio pipeline and model inference:
```bash
python verify_pipeline.py
```

### 2. Run End-to-End API Test
To test the actual running server with a real generated MP3 file:
```bash
# Ensure server is running in another terminal first!
python test_api.py
```

### 3. cURL Command
```bash
curl -X POST http://127.0.0.1:8000/api/voice-detection \
  -H "x-api-key: your-secret-key-123" \
  -H "Content-Type: application/json" \
  -d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
  }'
```

---

## 📂 Project Structure

```text
voice-detector/
├── app/
│   ├── main.py       # API Entry point & Routes
│   ├── infer.py      # Model Inference Logic (XLS-R + Classifier)
│   ├── audio.py      # Audio Normalization (Base64 -> 16kHz WAV)
│   └── auth.py       # Utilities
├── model/            # Model weights storage
├── requirements.txt  # Python dependencies
├── .env              # Config keys
├── verify_pipeline.py# System health check script
└── test_api.py       # Live API integration test
```

---

## 🧠 Model Logic (How it works)

1.  **Input**: Takes Base64 MP3.
2.  **Normalization**: Converts to **16,000Hz Mono WAV**.
3.  **Encoder**: Feeds audio into **Wav2Vec2-XLS-R-53** to get a 1024-dimensional embedding.
4.  **Feature Extraction**: Calculates **Pitch Variance** to detect robotic flatness.
5.  **Classifier**: A linear layer combines `[Embedding (1024) + Pitch (1)]` to predict `AI_GENERATED` or `HUMAN`.
