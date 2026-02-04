import yaml
import numpy as np
import os
import sys

# Add src to path if needed, or rely on root execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.audio import load_audio, to_tensor
from src.components.feature_extractor import FeatureExtractor
from src.components.rule_based import RuleBasedDetector
from src.components.model_wrapper import ModelWrapper
from src.utils.compatibility import apply_patches

# Apply dependency patches immediately
apply_patches()

class VoicePipeline:
    def __init__(self, config_path: str = "config/hparams.yaml"):
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.rule_detector = RuleBasedDetector(self.config.get("rules", {}))
        self.model_wrapper = ModelWrapper(self.config.get("model", {}))
        self.model_wrapper.config["vad"] = self.config.get("vad", {}) # Pass VAD config if separate
        self.model_wrapper.load_vad() # Ensure VAD loaded

    def _load_config(self, path: str) -> dict:
        if not os.path.exists(path):
            # Fallback default if config missing
            print(f"Config not found at {path}, using defaults.")
            return {}
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def process(self, audio_bytes: bytes) -> dict:
        """
        Process audio bytes and return classification result.
        """
        try:
            # 1. Load Audio
            audio_array, sr = load_audio(audio_bytes)
            
            # 2. Extract Features
            features = self.feature_extractor.extract(audio_array, sr)
            
            # 3. Rule-Based Check
            rule_label, rule_prob, rule_expl = self.rule_detector.predict(features)
            
            # 4. Model Prediction
            # Convert to tensor for model
            audio_tensor = to_tensor(audio_array)
            model_prob = self.model_wrapper.predict(audio_tensor, sr)
            
            # 5. Ensemble Logic
            # If Model is very confident, trust it.
            # If Model is unsure, check Rules.
            
            # Weights from config
            w_model = self.config.get("pipeline", {}).get("weights", {}).get("model", 0.7)
            w_rules = self.config.get("pipeline", {}).get("weights", {}).get("rules", 0.3)
            
            # Normalize rule prob (0.55/0.65 are arbitrary from reference, let's map to 0-1)
            # If HUMAN (0.55) -> 0.2? If AI (0.65) -> 0.8?
            # Let's just use the raw prob from rule detector if it makes sense, 
            # but rule detector retuns 0.65 for AI... that's low confidence.
            # Let's map "AI_GENERATED" to 0.9 and "HUMAN" to 0.1 for the sake of weighted average
            rule_score = 0.9 if rule_label == "AI_GENERATED" else 0.1
            
            combined_score = (model_prob * w_model) + (rule_score * w_rules)
            
            # Thresholds
            thresh_ai = self.config.get("pipeline", {}).get("thresholds", {}).get("ai_generated", 0.70)
            
            if combined_score >= thresh_ai:
                final_label = "AI_GENERATED"
                explanation = f"Detected synthetic patterns (Model: {model_prob:.2f}, Rules: {rule_label})"
            else:
                final_label = "HUMAN"
                explanation = f"Natural speech patterns (Model: {model_prob:.2f}, Rules: {rule_label})"
                
            return {
                "classification": final_label,
                "confidenceScore": float(combined_score),
                "explanation": explanation,
                "details": {
                    "model_probability": float(model_prob),
                    "rule_classification": rule_label,
                    "features": features # Optional: return features for debug
                }
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "error": str(e)
            }
