# Whisper + VAD 
Goal is to make a light and fast (as much as possible) implementation of Speech-to-Text system. With docker.
# Requirements
Python 3.11 (because torchaudio support it only up to 3.11 currently)
PyTorch==2.51.1 (need to separate build for CPU and GPU)
pyaudio (for recording)
onnxruntime (for fast VAD, have to separate GPU/CPU builds)
