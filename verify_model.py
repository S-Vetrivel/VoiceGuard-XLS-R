
import torch
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import numpy as np

def verify_model():
    model_name = "mo-thecreator/Deepfake-audio-detection"
    print(f"Loading {model_name}...")
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        model = AutoModelForAudioClassification.from_pretrained(model_name)
        print("Model loaded successfully!")
        
        print("Labels:", model.config.id2label)
        
        # Create dummy audio (1 second of silence/noise)
        # 16000 Hz
        dummy_audio = np.random.uniform(-1, 1, 16000)
        
        inputs = feature_extractor(dummy_audio, sampling_rate=16000, return_tensors="pt")
        
        with torch.no_grad():
            logits = model(**inputs).logits
        
        print("Logits:", logits)
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        print("Predicted Label:", model.config.id2label[predicted_class_id])
        
    except Exception as e:
        print(f"Failed to load/run model: {e}")

if __name__ == "__main__":
    verify_model()
