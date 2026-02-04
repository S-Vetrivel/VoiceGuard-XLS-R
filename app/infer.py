import torch
import torch.nn as nn
import os
import numpy as np
import librosa
import time
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
from dotenv import load_dotenv

load_dotenv()

class VoiceClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Deepfake Detection model on {self.device}...")
        
        # Load Fine-Tuned Deepfake Detection Model
        self.model_name = "mo-thecreator/Deepfake-audio-detection"
        
        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model {self.model_name} loaded successfully.")
            # Labels: {0: 'fake', 1: 'real'} usually for this model
            print(f"Labels: {self.model.config.id2label}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def predict(self, waveform: torch.Tensor):
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # 1. Preprocess Audio
            # Waveform is [1, T] Tensor. Convert to numpy [T]
            wav_np = waveform.squeeze().cpu().numpy()
            
            # Ensure we send it as a list or numpy array to the extractor
            inputs = self.feature_extractor(
                wav_np, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            t0 = time.time()
            
            # 2. Model Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # Logic for this specific model:
                # Label 0: 'fake' (AI)
                # Label 1: 'real' (Human)
                prob_fake = probs[0][0].item()
                prob_real = probs[0][1].item()
                
            t1 = time.time()
            print(f"DEBUG: Inference took {t1 - t0:.3f}s. probs: {probs}")

            # 3. Pitch Analysis (for explanation)
            # Use librosa for pitch tracking (fast approximation)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                wav_np, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=16000,
                frame_length=2048
            )
            f0 = f0[~np.isnan(f0)]
            pitch_var = np.std(f0) if len(f0) > 0 else 0.0
            
            t2 = time.time()
            print(f"DEBUG: Pitch Detection took {t2 - t1:.3f}s. Variance: {pitch_var}")

            # 4. Final Classification Logic
            # Deepfake model is the authority
            if prob_fake > prob_real:
                prediction = "AI_GENERATED"
                confidence = prob_fake
                prob_ai = prob_fake
            else:
                prediction = "HUMAN"
                confidence = prob_real
                prob_ai = prob_fake

            # Construct Explanation
            if prediction == "AI_GENERATED":
                if pitch_var < 20.0:
                    explanation = f"Deepfake model reported {confidence*100:.1f}% confidence. Detected unnatural pitch consistency (Variance: {pitch_var:.1f})."
                else:
                    explanation = f"Deepfake model reported {confidence*100:.1f}% confidence. Detected digital artifacts characteristic of AI synthesis."
            else:
                if pitch_var > 20.0:
                    explanation = f"Deepfake model reported {confidence*100:.1f}% confidence. Natural prosody and high pitch variance detected."
                else:
                    explanation = f"Deepfake model reported {confidence*100:.1f}% confidence. Audio classified as human despite low pitch variance."

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
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
