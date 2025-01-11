# Whisper + VAD 
Goal is to make a light and fast (as much as possible) implementation of Speech-to-Text system. With docker.
## Presequence

You need to get models first. It's expected 

Project uses [Silero VAD](https://github.com/snakers4/silero-vad) and OpenAI Whisper from [Hugginface Transformers](https://github.com/huggingface/transformers). So you need both of them.

### Silero VAD

You can download it from torch hub with `download_vad.py` script (but must install `torch` first!). It will automatically download and copy onnx file into `whisper_vad/model/vad` directory. Alternatively, you can symlink it:

``ln -s ~/.cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/data/silero_vad_16k_op15.onnx whisper_vad/model/vad/model.onnx``

### Whisper

To download Whisper from Hugginface Hub run 

``git clone https://huggingface.co/openai/whisper-large-v3-turbo``

It'll download it into `whisper-large-v3-turbo` directory. You can move its content into `whisper_vad/model/whisper` directory or you can use symlink instead

``ln -s /mnt/dev/models/whisper-large-v3-turbo app/model/whisper``

## Installation
### On host machine

1. Copy repository

```
git clone https://github.com/Drakrig/whisper-chunker.git
cd whisper-chunker
```

2. Install dependencies

```pip install -r requirements.txt```

3. Go into main directory and run `main.py`

```
cd whisper_vad
python3 main.py
```

### With docker compose

1. Install docker.
2. If you use symlinks to point to models, you have 2 options:

Create `.env` file

```
touch .env
nano .env
```
 with following content:

```
VAD_DIR=/path/to/onnx/model/dir/
WHISPER_DIR=/path/to/whisper/model/dir/
```

This is a good practice so it's recommended to use it.

Alternatively, in compose.yml, change the volumes section:

```
volumes:
    - "./whisper_vad:/app/"
    - "${VAD_DIR}:/app/model/vad/"
    - "${WHISPER_DIR}:/app/model/whisper/"
```

to

```
volumes:
    - "./whisper_vad:/app/"
    - "/path/to/onnx/model/dir/:/app/model/vad/"
    - "/path/to/whisper/model/dir/:/app/model/whisper/"
```

3. If you want to run whisper on GPU run (CUDA must be installed).

```docker compose -f compose.yml -f compose.gpu.yml up```

For CPU only run

```docker compose up```

It automatically build and launch container. The recognized text will be displayed in console as 

``Transcription: {text}``