
import torch
import torchaudio
import numpy as np
from speechbrain.inference.VAD import VAD

def verify_vad():
    model_source = "speechbrain/vad-crdnn-libriparty"
    print(f"Loading VAD model: {model_source}...")
    
    try:
        # Load VAD
        vad_model = VAD.from_hparams(
            source=model_source,
            savedir="tmp_vad_model",
            run_opts={"device": "cpu"} # Force CPU for verification
        )
        print("VAD Model loaded successfully!")
        
        # Create dummy audio (random noise + silence + random noise)
        sr = 16000
        duration = 5 # seconds
        t = np.linspace(0, duration, int(sr * duration))
        
        # 1 sec noise, 2 sec silence, 2 sec noise
        audio = np.random.uniform(-0.1, 0.1, int(sr * 1))
        audio = np.concatenate([audio, np.zeros(int(sr * 2))])
        audio = np.concatenate([audio, np.random.uniform(-0.1, 0.1, int(sr * 2))])
        
        # Convert to tensor path not needed if we can process tensor
        # SpeechBrain VAD usually expects a file path, but let's check input flexibility
        # For this test, save to a temp file
        import soundfile as sf
        sf.write('test_vad.wav', audio, sr)
        
        print("Processing test_vad.wav...")
        # Boundaries usually returns a tensor of [start, end]
        boundaries = vad_model.get_speech_segments("test_vad.wav")
        print(f"Speech Segments found: \n{boundaries}")
        
        # Check if it filtered the silence
        print("Verification complete.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_vad()
