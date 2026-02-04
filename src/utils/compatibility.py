import torchaudio

def apply_patches():
    """
    Apply compatibility patches for dependencies.
    """
    # SpeechBrain compatibility fix for torchaudio >= 2.1
    # Ensures list_audio_backends exists
    if not hasattr(torchaudio, "list_audio_backends"):
        def _list_audio_backends():
            return ["soundfile"]
        torchaudio.list_audio_backends = _list_audio_backends

    # SpeechBrain (<=1.0.3) passes 'use_auth_token' which was removed in huggingface_hub >= 0.23.0
    # Patch huggingface_hub.hf_hub_download to remap the argument
    import huggingface_hub
    from huggingface_hub import utils as hf_utils
    
    _original_hf_hub_download = huggingface_hub.hf_hub_download

    def _patched_hf_hub_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            # Remap to 'token' or remove if redundant (hf_hub_download handles 'token')
            token_val = kwargs.pop("use_auth_token")
            # Only set token if not already present
            if "token" not in kwargs:
                kwargs["token"] = token_val
        return _original_hf_hub_download(*args, **kwargs)

    huggingface_hub.hf_hub_download = _patched_hf_hub_download

