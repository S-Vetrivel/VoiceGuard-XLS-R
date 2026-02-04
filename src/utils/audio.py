import io
import librosa
import numpy as np
import torch
import torchaudio
import soundfile as sf

def load_audio(audio_bytes: bytes, target_sr: int = 16000, max_duration: int = 5) -> tuple[np.ndarray, int]:
    """
    Load audio from bytes, resample if necessary, and truncate/pad.
    Returns (audio_array, sample_rate).
    """
    try:
        # Load using librosa (handles various formats via soundfile/audioread)
        # mono=True mixes down to mono
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=target_sr, mono=True)
        
        # Truncate
        max_samples = int(target_sr * max_duration)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            
        return audio, sr
    except Exception as e:
        raise ValueError(f"Failed to load audio: {e}")

def to_tensor(audio_array: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch tensor."""
    return torch.tensor(audio_array).float()
