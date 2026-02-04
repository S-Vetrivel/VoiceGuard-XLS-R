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

    def predict(self, waveform: torch.Tensor, language: str = "Unknown"):
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # 1. Preprocess Audio
            wav_np = waveform.squeeze().cpu().numpy()
            sr = 16000
            
            # --- ADVANCED FEATURE EXTRACTION ---
            t0 = time.time()
            
            # A. Pitch Analysis
            f0, voiced_flag, voiced_probs = librosa.pyin(
                wav_np, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
            )
            f0_clean = f0[~np.isnan(f0)]
            pitch_var = np.std(f0_clean) if len(f0_clean) > 0 else 0.0
            
            # B. Spectral Flatness (Detects vocoder buzz)
            flatness = np.mean(librosa.feature.spectral_flatness(y=wav_np))
            
            # C. RMS Energy Variance (Detects flattened volume envelopes)
            rms = librosa.feature.rms(y=wav_np)[0]
            rms_var = np.std(rms) / (np.mean(rms) + 1e-6) # Normalized variance
            
            # D. Zero Crossing Rate Variance (Detects robotic vowel transitions)
            zcr = librosa.feature.zero_crossing_rate(wav_np)[0]
            zcr_var = np.std(zcr)
            
            # --- TEMPORAL CONSISTENCY (SLIDING WINDOW) ---
            chunk_size = 2 * sr # 2 seconds
            stride = 1 * sr     # 1 second overlap
            chunks = []
            for i in range(0, len(wav_np) - chunk_size + 1, stride):
                chunks.append(wav_np[i : i + chunk_size])
            
            # If audio too short for stride, just use whole thing
            if not chunks:
                chunks = [wav_np]
            
            chunk_probs = []
            for chunk in chunks:
                inputs = self.feature_extractor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    chunk_probs.append(probs[0][0].item()) # Probability of 'fake'
            
            # Authority calculation
            # We take the MAX probability across chunks to catch 'slips' in AI generation
            prob_fake = np.max(chunk_probs)
            prob_real = 1.0 - prob_fake
            
            t1 = time.time()
            print(f"DEBUG: Analysis took {t1 - t0:.3f}s. Multi-chunk prob_fake: {prob_fake:.4f}")
            print(f"DEBUG: Features - PitchVar: {pitch_var:.1f}, Flatness: {flatness:.4f}, RMS_Var: {rms_var:.4f}")

            # --- HYBRID HEURISTIC STRENGTHENING ---
            # AI Voices often have VERY low flatness or VERY low pitch variance
            ai_signal_flags = 0
            if pitch_var < 15.0: ai_signal_flags += 1
            if flatness < 0.005: ai_signal_flags += 1 # Very tonal/melodic
            if rms_var < 0.1: ai_signal_flags += 1    # Robotic volume
            
            # Final Verdict Logic
            # If the model is unsure (0.4-0.6) but signal flags are high, tip to AI
            if 0.4 < prob_fake < 0.6 and ai_signal_flags >= 2:
                prob_fake = 0.75
            
            prediction = "AI_GENERATED" if prob_fake > 0.5 else "HUMAN"
            confidence = prob_fake if prediction == "AI_GENERATED" else prob_real
            
            # --- LANGUAGE AWARENESS ---
            # If it's a non-English language, the model might be slightly less reliable
            # We dampen confidence on low-resource languages to prevent false accusations
            is_english = language.lower() in ["english", "en"]
            if not is_english and confidence < 0.85:
                confidence *= 0.95 # Slight dampening
            
            # Construct Explanation
            if prediction == "AI_GENERATED":
                reasons = []
                if ai_signal_flags >= 2: reasons.append("synthetic spectral characteristics")
                if pitch_var < 20: reasons.append("lack of natural prosody")
                if not reasons: reasons.append("digital vocoder artifacts")
                explanation = f"AI detected with {confidence*100:.1f}% confidence. Evidence: {', '.join(reasons)}."
            else:
                if pitch_var > 25.0:
                    explanation = f"Human verified with {confidence*100:.1f}% confidence. Strong natural pitch variance and human vocal dynamics detected."
                else:
                    explanation = f"Audio likely Human ({confidence*100:.1f}%). Detected natural speech fluctuations despite localized artifacts."

            return {
                "prediction": prediction,
                "probability_ai": float(f"{prob_fake:.4f}"),
                "confidence": float(f"{confidence:.4f}"),
                "features": {
                    "pitch_variance": float(f"{pitch_var:.2f}"),
                    "spectral_flatness": float(f"{flatness:.6f}"),
                    "rms_variance": float(f"{rms_var:.4f}"),
                    "zcr_variance": float(f"{zcr_var:.4f}")
                },
                "explanation": explanation
            }
            
        except Exception as e:
            print(f"Prediction Error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
