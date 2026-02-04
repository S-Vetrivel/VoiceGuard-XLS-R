import torch
import torch.nn as nn
import os
import numpy as np
import librosa
from transformers import Wav2Vec2Model
from dotenv import load_dotenv

load_dotenv()

class VoiceClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Wav2Vec2 model on {self.device}...")
        
        # Load Pretrained Wav2Vec2-XLS-R (Multilingual: 53 languages)
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.encoder.to(self.device)
        self.encoder.eval()
        
        # Freeze weights
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Linear Classifier (1024 embedding + 1 pitch var)
        # XLS-R-53 base outputs 1024 dimension features
        self.classifier = nn.Linear(1024 + 1, 1).to(self.device)
        # Initialize with dummy weights acting as a threshold for now
        # Logic: High pitch variance -> Human (negative logit?), Low -> AI (positive?)
        # For now we'll rely on training or manual setting. 
        # Let's set a bias that assumes Human (low prob AI) unless proven otherwise.
        nn.init.constant_(self.classifier.bias, -1.0)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        
        print("Model loaded successfully.")

    def extract_features(self, waveform: torch.Tensor):
        """
        waveform: [1, T] Tensor at 16kHz
        Returns: feature_vector [1, 769]
        """
        waveform = waveform.to(self.device)
        
        # 1. Wav2Vec2 Embedding
        with torch.no_grad():
            outputs = self.encoder(waveform)
            # last_hidden_state: [1, Sequence, 768]
            hidden_states = outputs.last_hidden_state
            # Mean Pooling -> [1, 768]
            embedding = torch.mean(hidden_states, dim=1)
            
        # 2. Pitch Variance
        # Move to CPU for numpy/librosa ops
        wav_np = waveform.squeeze().cpu().numpy()
        
        # Use librosa for pitch tracking (fast approximation)
        # fmin/fmax for human speech range
        f0, voiced_flag, voiced_probs = librosa.pyin(
            wav_np, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=16000,
            frame_length=2048
        )
        
        # Filter NaNs
        f0 = f0[~np.isnan(f0)]
        
        if len(f0) > 0:
            pitch_std = np.std(f0)
            # Normalize? Let's just keep raw for now, or log scale
            pitch_var = pitch_std
        else:
            pitch_var = 0.0
            
        # Combine
        pitch_feature = torch.tensor([[pitch_var]], device=self.device, dtype=torch.float32)
        
        # Concatenate [1, 768] + [1, 1] -> [1, 769]
        features = torch.cat((embedding, pitch_feature), dim=1)
        return features, pitch_var

    def predict(self, waveform: torch.Tensor):
        if self.encoder is None:
            return {"error": "Model not loaded"}
        
        try:
            features, pitch_var = self.extract_features(waveform)
            
            with torch.no_grad():
                logits = self.classifier(features)
                prob_ai = torch.sigmoid(logits).item()
                
            # Explainability
            # CONFIDENCE = max(p, 1-p)
            confidence = max(prob_ai, 1 - prob_ai)
            
            # Strict Classification Labels
            prediction = "AI_GENERATED" if prob_ai > 0.5 else "HUMAN"
            
            explanation = "High pitch variance and natural prosody detected." if pitch_var > 20.0 else "Unnatural pitch consistency and robotic speech patterns detected."
            
            return {
                "prediction": prediction,
                "probability_ai": float(f"{prob_ai:.4f}"),
                "confidence": float(f"{confidence:.4f}"),
                "features": {
                    "pitch_variance": float(f"{pitch_var:.2f}")
                },
                "explanation": explanation
            }
            
        except Exception as e:
            print(f"Prediction Error: {e}")
            return {"error": str(e)}
