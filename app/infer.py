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

    def calculate_snr(self, audio_np):
        """
        Estimate Signal-to-Noise Ratio (SNR) in dB.
        Assumes the quietest 10% of frames represent the noise floor.
        """
        try:
            # Frame-based RMS energy
            rms = librosa.feature.rms(y=audio_np)[0]
            if len(rms) < 10: return 50.0 # Too short, assume clean
            
            # Sort RMS values to find noise floor
            sorted_rms = np.sort(rms)
            noise_len = max(1, int(0.1 * len(rms)))
            noise_floor_rms = np.mean(sorted_rms[:noise_len]) + 1e-9
            
            # Signal RMS (approximate as top 50% energy average)
            signal_len = max(1, int(0.5 * len(rms)))
            signal_rms = np.mean(sorted_rms[-signal_len:])
            
            snr = 20 * np.log10(signal_rms / noise_floor_rms)
            return snr
        except Exception:
            return 30.0 # Default to decent SNR if calculation fails

    def predict(self, waveform: torch.Tensor, language: str = "Unknown"):
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # 1. Preprocess Audio
            wav_np = waveform.squeeze().cpu().numpy()
            sr = 16000
            
            t0 = time.time()
            
            # --- SIGNAL QUALITY CHECKS ---
            snr_db = self.calculate_snr(wav_np)
            
            # --- ADVANCED FEATURE EXTRACTION ---
            # A. Pitch Analysis
            f0, voiced_flag, voiced_probs = librosa.pyin(
                wav_np, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
            )
            f0_clean = f0[~np.isnan(f0)]
            pitch_var = np.std(f0_clean) if len(f0_clean) > 0 else 0.0
            
            # B. Spectral Flatness
            flatness = np.mean(librosa.feature.spectral_flatness(y=wav_np))
            
            # C. RMS Energy Variance
            rms = librosa.feature.rms(y=wav_np)[0]
            rms_var = np.std(rms) / (np.mean(rms) + 1e-6)
            
            # D. Liveness (Pause) Detection
            # Count distinct silent intervals (>0.1s)
            silent_intervals = librosa.effects.split(wav_np, top_db=20, frame_length=2048, hop_length=512)
            num_pauses = 0
            if len(silent_intervals) > 1:
                # Calculate gaps between speech segments
                for i in range(len(silent_intervals)-1):
                    gap_samples = silent_intervals[i+1][0] - silent_intervals[i][1]
                    if gap_samples > sr * 0.1: # >100ms
                         num_pauses += 1
            
            # --- TEMPORAL CONSISTENCY ---
            chunk_size = 2 * sr 
            stride = 1 * sr     
            chunks = []
            for i in range(0, len(wav_np) - chunk_size + 1, stride):
                chunks.append(wav_np[i : i + chunk_size])
            if not chunks: chunks = [wav_np]
            
            chunk_probs = []
            for chunk in chunks:
                inputs = self.feature_extractor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    chunk_probs.append(probs[0][0].item()) # Prob fake
            
            # Initial Raw Confidence (Max across chunks)
            prob_fake = np.max(chunk_probs)
            
            t1 = time.time()
            print(f"DEBUG: Analysis took {t1 - t0:.3f}s. Raw prob_fake: {prob_fake:.4f}")
            print(f"DEBUG: Features - SNR: {snr_db:.1f}dB, Pauses: {num_pauses}, PitchVar: {pitch_var:.1f}, Flatness: {flatness:.4f}")

            # --- CONSERVATIVE CONSENSUS LOGIC ---
            
            # 1. Initialize Flags (Relaxed thresholds)
            ai_flags = []
            human_flags = []
            
            # AI Indicators
            if pitch_var < 10.0: ai_flags.append("Low pitch variance") # Relaxed from 15
            if flatness < 0.002: ai_flags.append("Unnatural spectral flatness") # Relaxed from 0.005
            if rms_var < 0.1: ai_flags.append("Robotic volume consistency")
            
            # Human Indicators (VETO Power)
            if snr_db < 15.0: human_flags.append("High Background Noise")
            if num_pauses >= 2: human_flags.append("Natural breathing pauses")
            if pitch_var > 35.0: human_flags.append("High expressive variation")
            
            # 2. Apply Penalties / Vetoes
            confidence_penalty = 1.0
            
            # VETO 1: NOISE
            # If noisy, the model's "Fake" detection is untrustworthy. Cap it.
            if snr_db < 15.0:
                print("DEBUG: Low SNR detected. Applying penalty.")
                confidence_penalty *= 0.6 # Reduce confidence by 40%
            
            # VETO 2: LIVENESS
            if num_pauses >= 2 and prob_fake < 0.95:
                 print("DEBUG: Natural pauses detected. Applying penalty.")
                 confidence_penalty *= 0.8 # Reduce confidence by 20%
                 
            # Apply penalty to the probability of being fake
            prob_fake_adjusted = prob_fake * confidence_penalty
            
            # --- LANGUAGE AWARENESS ---
            is_english = language.lower() in ["english", "en"]
            
            # 3. Final Decision
            # We demand HIGHER evidence for AI (Conservatism)
            
            # Base threshold
            threshold = 0.65 
            
            # Dynamic Thresholding based on Heuristics
            if len(ai_flags) >= 2:
                # Strong heuristic evidence (e.g. robotic pitch + flat spectrum)
                # We lower the bar for the model
                threshold = 0.50
            elif len(ai_flags) == 1:
                # Some heuristic evidence
                threshold = 0.60
            else:
                # ZERO heuristic evidence (Pitch/Flatness look human)
                # The model is alone in its accusation.
                if not is_english:
                    # Foreign language + No Heuristics = FALSE POSITIVE likely.
                    # We force Human verdict unless we want to be extremely risky.
                    # Current decision: Force Human to protect against bias.
                    print("DEBUG: Non-English audio with NO heuristic AI flags. Forcing Human verdict.")
                    prob_fake_adjusted = 0.0 
                else:
                    # English + No Heuristics. 
                    # Model must be overwhelmingly confident (>98%) to override heuristics.
                    threshold = 0.98

            if prob_fake_adjusted > threshold:
                prediction = "AI_GENERATED"
                confidence = prob_fake_adjusted
            else:
                prediction = "HUMAN"
                confidence = 1.0 - prob_fake_adjusted
            
            # 4. Language Awareness Dampening (for the resulting score)
            if prediction == "AI_GENERATED" and not is_english:
                 confidence *= 0.9 # Extra caution for non-English
            
            # Construct Explanation
            if prediction == "AI_GENERATED":
                reasons = ai_flags
                if not reasons: reasons.append("high confidence from deepfake classifier")
                explanation = f"AI detected ({confidence*100:.1f}%). Indicators: {', '.join(reasons)}."
            else:
                reasons = human_flags
                if not reasons: reasons.append("insufficient evidence of synthesis")
                explanation = f"Verified Human ({confidence*100:.1f}%). Evidence: {', '.join(reasons)}."

            return {
                "prediction": prediction,
                "probability_ai": float(f"{prob_fake_adjusted:.4f}"),
                "confidence": float(f"{confidence:.4f}"),
                "features": {
                    "pitch_variance": float(f"{pitch_var:.2f}"),
                    "snr_db": float(f"{snr_db:.1f}"),
                    "pauses": num_pauses
                },
                "explanation": explanation
            }
            
        except Exception as e:
            print(f"Prediction Error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
