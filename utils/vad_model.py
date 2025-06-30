import os
from .vad_utils.utils_vad import init_jit_model, OnnxWrapper, get_speech_timestamps, read_audio
from .vad_utils.create_segments import hierarchical_merge_vad_segments, sequential_merge_vad_segments
import torch

class VAD():
    def __init__(self, model):
        self.model = model

    def get_speech_timestamps(self, wav, **kwargs):
        return get_speech_timestamps(wav, self.model, **kwargs)
    
    def merge_segments(self, timestamps, max_pause=0.0, max_seg_len=30.0, min_seg_len=0.0, mode="hierarchical"):
        if mode == "hierarchical":
            merged = hierarchical_merge_vad_segments(timestamps, max_pause, max_seg_len, min_seg_len)
        elif mode == "sequential":
            merged = sequential_merge_vad_segments(timestamps, max_pause, max_seg_len)
        else:
            raise ValueError(f"merge segments mode should be hierarchical or sequential, got {mode}")
        return merged

def load_vad(model_dir, onnx=False, device="cuda"):
    model_name = 'silero_vad.onnx' if onnx else 'silero_vad.jit'
    model_file_path = os.path.join(model_dir, model_name)

    assert os.path.exists(model_file_path), f"VAD model does not exist: {model_file_path}"

    if onnx:
        assert device == "cpu", "ONNX was selected but device was set to GPU"
        model = OnnxWrapper(model_file_path, force_onnx_cpu=True)
    else:
        model = init_jit_model(model_file_path)
        model = model.to(device=device)
    
    model = VAD(model)
    return model

def load_audio(filepath, sampling_rate=16000, device="cuda"):
    return read_audio(filepath, sampling_rate=sampling_rate, device=device)

