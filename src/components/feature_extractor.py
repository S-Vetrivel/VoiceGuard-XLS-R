import librosa
import numpy as np

class FeatureExtractor:
    def extract(self, audio: np.ndarray, sr: int) -> dict:
        """
        Extract handcrafted features for rule-based detection.
        Ported from AI-Generated-Voice-Detection reference.
        """
        features = {}

        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        # Filter out zero pitches
        pitch_values = pitches[pitches > 0]
        
        features["pitch_mean"] = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
        features["pitch_std"] = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0

        # MFCCs (13 coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfcc, axis=1)
        for i, val in enumerate(mfcc_means):
            features[f"mfcc_{i+1}"] = float(val)

        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features["spectral_centroid_mean"] = float(np.mean(centroid))

        # Energy variation (RMS)
        rms = librosa.feature.rms(y=audio)
        features["rms_std"] = float(np.std(rms))

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        features["zcr_mean"] = float(np.mean(zcr))

        return features
