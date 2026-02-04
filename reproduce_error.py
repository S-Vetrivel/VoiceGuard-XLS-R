import torch
from app.audio import process_audio
from pydub import AudioSegment
import numpy as np
import io
import base64

def generate_mp3_base64():
    """Generates a valid 1-second MP3 sine wave and returns base64 string."""
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * 440 * t)
    x_int = (x * 32767).astype(np.int16)
    audio = AudioSegment(
        x_int.tobytes(), 
        frame_rate=sr,
        sample_width=2, 
        channels=1
    )
    mp3_io = io.BytesIO()
    audio.export(mp3_io, format="mp3")
    return base64.b64encode(mp3_io.getvalue()).decode('utf-8')

def test_reproduce():
    print("--- 🔍 Testing Valid Audio Processing ---")
    
    # 1. Generate Valid MP3
    valid_b64 = generate_mp3_base64()
    print(f"Generated Valid B64 Length: {len(valid_b64)}")
    
    try:
        # 2. Test Processing
        waveform = process_audio(valid_b64)
        print("\n✅ Success with Valid MP3!")
        print(f"Waveform shape: {waveform.shape}")
        
    except Exception as e:
        print("\n❌ Failed with Valid MP3!")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_reproduce()
