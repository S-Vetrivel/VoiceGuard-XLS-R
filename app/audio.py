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
                    audio_segment = AudioSegment.from_file(input_data)
                else:
                    raise FileNotFoundError
            except:
                # Assume Base64 string if file load fails
                # 1. Clean up "data:audio/mp3;base64," prefix if present
                clean_b64 = input_data
                if "," in clean_b64:
                    clean_b64 = clean_b64.split(",", 1)[1]
                
                # 2. Remove whitespace/newlines which break some decoders
                clean_b64 = clean_b64.strip().replace("\n", "").replace(" ", "")
                
                decoded_bytes = base64.b64decode(clean_b64)
                audio_segment = AudioSegment.from_file(io.BytesIO(decoded_bytes))
        elif isinstance(input_data, bytes):
            audio_segment = AudioSegment.from_file(io.BytesIO(input_data))
        else:
            raise ValueError("Unsupported input type. Expected: str (path/base64) or bytes.")
            
    except Exception as e:
        raise ValueError(f"Failed to load audio: {e}")

    # 2. Resample to 16kHz
    if audio_segment.frame_rate != TARGET_SR:
        audio_segment = audio_segment.set_frame_rate(TARGET_SR)

    # 3. Convert to Mono
    if audio_segment.channels > 1:
        audio_segment = audio_segment.set_channels(1)

    # 4. Convert to Numpy Array (float32)
    # pydub audio is int16 or int32 generally, we want float32 [-1, 1]
    samples = np.array(audio_segment.get_array_of_samples())
    
    if audio_segment.sample_width == 2:
        samples = samples.astype(np.float32) / 32768.0
    elif audio_segment.sample_width == 4:
        samples = samples.astype(np.float32) / 2147483648.0
    else:
        # Fallback for 8-bit?
        samples = samples.astype(np.float32) / 128.0

    # 5. Convert to Torch Tensor [1, T]
    waveform = torch.tensor(samples).unsqueeze(0)
    
    return waveform
