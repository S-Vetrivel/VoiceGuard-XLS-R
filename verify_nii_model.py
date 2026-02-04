
import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

def verify_nii_model():
    model_id = "nii-yamagishilab/mms-300m-anti-deepfake"
    base_id = "facebook/mms-300m"
    
    print(f"Loading Feature Extractor from {base_id}...")
    try:
        # MMS uses Wav2Vec2FeatureExtractor
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_id)
        print("Feature Extractor loaded.")
        
        print(f"Loading Model from {model_id}...")
        model = AutoModelForAudioClassification.from_pretrained(model_id)
        print("Model loaded successfully!")
        
        # Check standard config
        print(f"Labels: {model.config.id2label}")
        
        # Test with dummy audio
        dummy_audio = np.random.uniform(-1, 1, 16000) # Random noise
        inputs = feature_extractor(dummy_audio, sampling_rate=16000, return_tensors="pt")
        
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            print(f"Dummy output probabilities: {probs}")
            predicted_id = torch.argmax(logits, dim=-1).item()
            label = model.config.id2label.get(predicted_id, str(predicted_id))
            print(f"Prediction: {label}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_nii_model()
