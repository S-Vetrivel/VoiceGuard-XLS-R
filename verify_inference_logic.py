
import os
import sys
import numpy as np
import io
from pydub import AudioSegment

# Ensure app is in path
sys.path.append(os.getcwd())

from app.infer import VoiceClassifier
from app.audio import process_audio

def verify():
    classifier = VoiceClassifier()
    
    # Generate a valid sine wave MP3 in memory
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * 440 * t)
    x_int = (x * 32767).astype(np.int16)
    audio = AudioSegment(x_int.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    
    mp3_io = io.BytesIO()
    audio.export(mp3_io, format="mp3")
    mp3_bytes = mp3_io.getvalue()
    
    print(f"Generated test MP3 bytes: {len(mp3_bytes)} bytes")
    
    # Process audio raw bytes
    waveform = process_audio(mp3_bytes)
    
    # Predict
    result = classifier.predict(waveform)
    print("Result:", result)
    
if __name__ == "__main__":
    verify()
