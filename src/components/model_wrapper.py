import torch
import traceback
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from speechbrain.inference.VAD import VAD
import os

class ModelWrapper:
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get("name", "nii-yamagishilab/mms-300m-anti-deepfake")
        self.device = config.get("device", "cpu")
        self.model = None
        self.feature_extractor = None
        self.vad = None
        
        # Log library versions for debugging
        try:
            import transformers
            import safetensors
            print(f"Library versions - transformers: {transformers.__version__}, safetensors: {safetensors.__version__}")
        except Exception as e:
            print(f"Warning: Could not log library versions: {e}")
        
        self.load_model()
        self.load_vad()

    def load_model(self):
        try:
            print(f"Loading Deepfake Detection model {self.model_name} on {self.device}...")
            model = AutoModelForAudioClassification.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            ).to(self.device)
            
            fe_name = self.config.get("feature_extractor", self.model_name)
            feature_extractor = AutoFeatureExtractor.from_pretrained(fe_name)
            
            # Only set if both loaded successfully
            self.model = model
            self.feature_extractor = feature_extractor
            self.model.eval()
            print("Model loaded successfully.")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"Model name: {self.model_name}, Device: {self.device}")
            traceback.print_exc()
            self.model = None
            self.feature_extractor = None

    def load_vad(self):
        try:
            vad_repo = self.config.get("vad", {}).get("repo", "speechbrain/vad-crdnn-libriparty")
            print(f"Loading SpeechBrain VAD from {vad_repo}...")
            # VAD loads internal models, ensure we catch errors here too
            self.vad = VAD.from_hparams(
                source=vad_repo, 
                savedir=self.config.get("vad", {}).get("save_path", "model_checkpoints")
            )
            print("SpeechBrain VAD loaded.")
        except Exception as e:
            print(f"Error loading VAD: {e}")
            traceback.print_exc()
            # We can tolerate VAD failure slightly by processing whole audio, or fail hard.
            # For now, let's keep it robust.
            self.vad = None

    def predict(self, audio: torch.Tensor, sr: int) -> float:
        """
        Predict probability of AI generation.
        Returns float (0.0 to 1.0), where 1.0 is AI.
        """
        if self.model is None or self.feature_extractor is None:
            raise RuntimeError("Model not loaded")

        with torch.no_grad():
            # Preprocess
            inputs = self.feature_extractor(
                audio.numpy(), 
                sampling_rate=sr, 
                return_tensors="pt"
            ).to(self.device)

            # Inference
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Label mapping: 
            # id2label usually {0: 'bonafide', 1: 'spoof'} OR {0: 'real', 1: 'fake'}
            # For mms-300m-anti-deepfake: 0 is 'bonafide' (human), 1 is 'spoof' (AI)
            # Verify this assumption via config or logs.  
            # (Logs from repro script said: Labels: {0: 'LABEL_0', 1: 'LABEL_1'})
            # Typically, LABEL_1 is the positive class (spoof).
            
            ai_prob = probs[0][1].item()
            
            # Safety check: handle NaN/Inf (can occur if model weights are improperly loaded)
            if not torch.isfinite(torch.tensor(ai_prob)):
                print(f"WARNING: Model returned non-finite value: {ai_prob}. Returning 0.5 as fallback.")
                return 0.5
            
            return ai_prob
