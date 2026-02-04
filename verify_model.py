
import torch
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
import numpy as np

def check_model():
    model_name = "nii-yamagishilab/mms-300m-anti-deepfake"
    feature_extractor_name = "facebook/mms-300m"
    
    print(f"Verifying load for: {model_name}")
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(feature_extractor_name)
        model = AutoModelForAudioClassification.from_pretrained(model_name)
        print("Success! Model and Extractor loaded.")
        print(f"Classes: {model.config.id2label}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    check_model()
