import torch
import numpy as np
import io
import base64
import os
from pydub import AudioSegment
import librosa # Keep librosa for easy array handling if needed, or just use pydub + numpy

TARGET_SR = 16000

def process_audio(input_data) -> torch.Tensor:
    """
    Decodes audio from file path, bytes, or base64 string.
    Normalizes to 16kHz, Mono, and returns a Torch Tensor [1, T].
    """
    audio_segment = None

    # 1. Load Audio
    try:
        if isinstance(input_data, str):
            # Check if it's a file path
            try:
                if os.path.isfile(input_data):
                    print(f"DEBUG: Loading audio from file: {input_data}")
                    audio_segment = AudioSegment.from_file(input_data)
                else:
                    raise FileNotFoundError
            except:
                # Assume Base64 string if file load fails
                print("DEBUG: Processing input as Base64 string...")
                
                # 1. Clean up headers and whitespace
                clean_b64 = input_data
                if "," in clean_b64:
                    clean_b64 = clean_b64.split(",", 1)[1]
                clean_b64 = clean_b64.strip().replace("\n", "").replace(" ", "")
                
                # 2. Fix Padding
                missing_padding = len(clean_b64) % 4
                if missing_padding:
                    clean_b64 += '=' * (4 - missing_padding)
                
                print(f"DEBUG: Base64 string length: {len(clean_b64)}")
                
                try:
                    decoded_bytes = base64.b64decode(clean_b64)
                    print(f"DEBUG: Decoded bytes length: {len(decoded_bytes)}")
                    print(f"DEBUG: First 16 bytes: {decoded_bytes[:16].hex()}")
                    
                    # 3. Explicitly try MP3 first, then let pydub probe
                    try:
                        audio_segment = AudioSegment.from_file(io.BytesIO(decoded_bytes), format="mp3")
                    except Exception as mp3_err:
                        print(f"DEBUG: Explicit MP3 load failed ({mp3_err}), trying auto-detection...")
                        audio_segment = AudioSegment.from_file(io.BytesIO(decoded_bytes))
                        
                except Exception as b64_err:
                    print(f"ERROR: Base64 decode failed: {b64_err}")
                    raise ValueError(f"Invalid Base64 string: {b64_err}")
        elif isinstance(input_data, bytes):
            audio_segment = AudioSegment.from_file(io.BytesIO(input_data))
        else:
            raise ValueError("Unsupported input type. Expected: str (path/base64) or bytes.")
            
    except Exception as e:
        print(f"CRITICAL ERROR in process_audio: {e}")
        raise ValueError(f"Failed to load audio: {e}")

    # 1.5 Truncate to Max Duration (5 seconds) to prevent timeouts on CPU
    MAX_DURATION_MS = 5000
    if len(audio_segment) > MAX_DURATION_MS:
        print(f"DEBUG: Audio too long ({len(audio_segment)}ms). Truncating to {MAX_DURATION_MS}ms.")
        audio_segment = audio_segment[:MAX_DURATION_MS]

    # 2. Resample to 16kHz
    if audio_segment.frame_rate != TARGET_SR:
        audio_segment = audio_segment.set_frame_rate(TARGET_SR)

    # 3. Convert to Mono
    if audio_segment.channels > 1:
        audio_segment = audio_segment.set_channels(1)

    # 4. Convert to Numpy Array (float32)
    # pydub audio is int16 or int32 generally, we want float32 [-1, 1]
    samples = np.array(audio_segment.get_array_of_samples())
    print(f"DEBUG: Loaded samples array shape: {samples.shape}")
    
    if audio_segment.sample_width == 2:
        samples = samples.astype(np.float32) / 32768.0
    elif audio_segment.sample_width == 4:
        samples = samples.astype(np.float32) / 2147483648.0
    else:
        samples = samples.astype(np.float32) / 128.0

    # 5. Convert to Torch Tensor [1, T]
    waveform = torch.tensor(samples).unsqueeze(0)
    print(f"DEBUG: Output waveform tensor shape: {waveform.shape}")
    
    return waveform
