class RuleBasedDetector:
    def __init__(self, config: dict):
        self.config = config

    def predict(self, features: dict) -> tuple[str, float, str]:
        """
        Apply heuristic rules to features.
        Returns (label, confidence, explanation).
        """
        score = 0
        reasons = []

        # Rules ported from AI-Generated-Voice-Detection
        pitch_std_thresh = self.config.get("pitch_std_threshold", 50.0)
        spec_cent_thresh = self.config.get("spectral_centroid_threshold", 3000.0)
        rms_std_thresh = self.config.get("rms_std_threshold", 0.01)

        if features["pitch_std"] < pitch_std_thresh:
            score += 1
            reasons.append("Unnaturally stable pitch detected")

        if features["spectral_centroid_mean"] > spec_cent_thresh:
            score += 1
            reasons.append("Overly smooth spectral characteristics")

        if features["rms_std"] < rms_std_thresh:
            score += 1
            reasons.append("Low energy variation typical of synthetic speech")

        if score >= 2:
            return "AI_GENERATED", 0.65, "; ".join(reasons)

        return "HUMAN", 0.55, "Natural human-like speech dynamics observed"
