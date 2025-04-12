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
AUDIO_DEVICE_INDEX=device_index
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

## How to access microphone from Docker.

I only tried it on Linux so far. It might be tricky to do it on Windows. Best solution probably will be to use something with WSL, but it just a guess.

Compose file have all necessary preparations done. The only thing you need to adjust is `AUDIO_DEVICE_INDEX` value in `.env` file.

The most straightforward way to find your device index is to use `sounddevice` package. The best way would be to run it inside Docker container. To do so, add this lines to `whisper` service in compose file:
```
stdin_open: true
tty: true
```
then run

```
docker exec container_name /bin/bash
```
As you inside the container, start Python CLI
```
python3
```

and run the code below 

```
import sounddevice as sd

print(sd.query_devices())
```

You'll something like this:

```
  0 USB Audio: - (hw:0,0), ALSA (2 in, 8 out)
  1 USB Audio: #1 (hw:0,1), ALSA (2 in, 2 out)
  ...
* 32 default, ALSA (128 in, 128 out)
  33 dmix, ALSA (0 in, 2 out)
```

Default device seems not to work properly in Docker so you need to determine device, in which your microphone plugged in. After all of this, adjust `AUDIO_DEVICE_INDEX` according to your findings.