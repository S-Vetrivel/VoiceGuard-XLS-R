import os
import torch
import torchaudio
import librosa
import numpy as np
import time
import shutil
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
from speechbrain.inference.VAD import VAD
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

class VoiceClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Deepfake Detection model on {self.device}...")
        
        # Load MMS-300M Anti-Deepfake Model (XLS-R based)
        self.model_name = "nii-yamagishilab/mms-300m-anti-deepfake"
        self.feature_extractor_name = "facebook/mms-300m"
        
        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.feature_extractor_name)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model {self.model_name} loaded successfully (MMS Backbone).")
            # Labels: {0: 'fake', 1: 'real'} usually for this model
            print(f"Labels: {self.model.config.id2label}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

        # Load SpeechBrain VAD
        try:
            print("Loading SpeechBrain VAD...")
            self.vad_model = VAD.from_hparams(
                source="speechbrain/vad-crdnn-libriparty",
                savedir="tmp_vad_model",
                run_opts={"device": str(self.device)}
            )
            print("SpeechBrain VAD loaded.")
        except Exception as e:
            print(f"Error loading VAD: {e}")
            self.vad_model = None

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

    def apply_vad(self, wav_path):
        """
        Apply VAD to filter out silence/noise.
        Returns cleaned waveform (numpy) or original if failed/empty.
        """
        if self.vad_model is None:
            return None
        
        try:
            # Get speech segments
            boundaries = self.vad_model.get_speech_segments(wav_path)
            
            # If tensor, convert to list
            if isinstance(boundaries, torch.Tensor):
                boundaries = boundaries.cpu().numpy()
            
            # Load original audio
            wav, sr = librosa.load(wav_path, sr=16000)
            
            if len(boundaries) == 0:
                print("DEBUG: VAD found no speech. Using original.")
                return wav
            
            # Concatenate segments
            cleaned_wavs = []
            for start, end in boundaries:
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                if end_sample > len(wav): end_sample = len(wav)
                cleaned_wavs.append(wav[start_sample:end_sample])
            
            if not cleaned_wavs:
                return wav
                
            final_wav = np.concatenate(cleaned_wavs)
            print(f"DEBUG: VAD reduced audio from {len(wav)/sr:.2f}s to {len(final_wav)/sr:.2f}s")
            return final_wav
            
        except Exception as e:
            print(f"VAD Error: {e}")
            return None

    def predict(self, waveform: torch.Tensor, language: str = "Unknown"):
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # 1. Preprocess Audio
            wav_np = waveform.squeeze().cpu().numpy()
            sr = 16000
            
            # Save to temp file for VAD (SpeechBrain prefers files)
            tmp_file = "temp_vad_input.wav"
            sf.write(tmp_file, wav_np, sr)
            
            # --- STAGE 1: SPEECHBRAIN VAD ---
            t0 = time.time()
            vad_wav = self.apply_vad(tmp_file)
            
            # Use VAD audio if valid and not too short, else original
            if vad_wav is not None and len(vad_wav) > sr * 0.5:
                wav_for_analysis = vad_wav
            else:
                wav_for_analysis = wav_np
                
            # Signal Quality Checks (on original to capture noise floor)
            snr_db = self.calculate_snr(wav_np)
            
            # --- ADVANCED FEATURE EXTRACTION (on VAD audio) ---
            # A. Pitch Analysis
            f0, voiced_flag, voiced_probs = librosa.pyin(
                wav_for_analysis, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
            )
            f0_clean = f0[~np.isnan(f0)]
            pitch_var = np.std(f0_clean) if len(f0_clean) > 0 else 0.0
            
            # B. Spectral Flatness
            flatness = np.mean(librosa.feature.spectral_flatness(y=wav_for_analysis))
            
            # C. RMS Energy Variance
            rms = librosa.feature.rms(y=wav_for_analysis)[0]
            rms_var = np.std(rms) / (np.mean(rms) + 1e-6)
            
            # D. Liveness (Pause) Detection (Use original to detect gaps)
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
            # Use VAD audio for Deepfake Classification
            chunk_size = 2 * sr 
            stride = 1 * sr     
            chunks = []
            for i in range(0, len(wav_for_analysis) - chunk_size + 1, stride):
                chunks.append(wav_for_analysis[i : i + chunk_size])
            if not chunks: chunks = [wav_for_analysis]
            
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
            # We demand HIGHER evidence for AI (Conservatism) but trust MMS more.
            
            # Base threshold
            threshold = 0.60
            
            # Dynamic Thresholding based on Heuristics
            if len(ai_flags) >= 2:
                # Strong heuristic evidence (e.g. robotic pitch + flat spectrum)
                threshold = 0.50
            elif len(ai_flags) == 1:
                # Some heuristic evidence
                threshold = 0.55
            else:
                # ZERO heuristic evidence (Pitch/Flatness look human)
                # The model is alone in its accusation.
                if not is_english:
                    # Foreign language + No Heuristics.
                    # MMS is multilingual, so we don't zero it out, but we require HIGH confidence.
                    print("DEBUG: Non-English audio with NO heuristic AI flags. Requiring high MMS confidence.")
                    threshold = 0.90 # High bar, but possible (unlike previous 0.0 force)
                else:
                    # English + No Heuristics. 
                    threshold = 0.98

            if prob_fake_adjusted > threshold:
                prediction = "AI_GENERATED"
                confidence = prob_fake_adjusted
            else:
                prediction = "HUMAN"
                confidence = 1.0 - prob_fake_adjusted
            
            # 4. Language Awareness Dampening (MMS is robust, lesser dampening)
            if prediction == "AI_GENERATED" and not is_english:
                 confidence *= 0.95 # Slight caution only
            
            # Construct Explanation
            if prediction == "AI_GENERATED":
                reasons = ai_flags
                if not reasons: reasons.append("high confidence from MMS (XLS-R) classifier")
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
