import torch
from pathlib import Path
import shutil

_, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                model='silero_vad',
                                onnx=True)

models_dir = Path(torch.hub.get_dir()).joinpath("snakers4_silero-vad_master/src/silero_vad/data/")

if not Path("whisper_vad/model/vad/model.onnx").exists():
    shutil.copy(models_dir.joinpath("silero_vad_16k_op15.onnx"), "whisper_vad/model/vad/model.onnx")
else:
    print("Model already exists")