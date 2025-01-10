from queue import Queue
import pyaudio
import numpy as np
from threading import Thread
from logging import getLogger, basicConfig, INFO

logger = getLogger(__name__)
basicConfig(level=INFO)

class Recorder():
    def __init__(self,
                 output_queue:Queue,
                 sample_rate=16000,
                 frame_duration_ms=32):
        self.output_queue = output_queue
        self.sr = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.running = False
    
    def _stream_audio(self):
        while self.running:
            data = self.audio_stream.read(self.frame_size)
            data = np.frombuffer(data, dtype=np.float32)
            self.output_queue.put(data)
    
    def start_recording(self, device_index=None):
        self.audio = pyaudio.PyAudio()
        self.audio_stream = self.audio.open(format=pyaudio.paFloat32,
                                       channels=1,
                                       rate=self.sr,
                                       input=True,
                                       frames_per_buffer=self.frame_size,
                                       input_device_index=device_index)
        self.running = True
        self.stream_thread = Thread(target=self._stream_audio)
        self.stream_thread.start()
    
    def stop_recording(self):
        logger.info("Stopping recording")
        self.running = False
        self.stream_thread.join()
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.audio.terminate()
        logger.info("Recording stopped")

if __name__ == "__main__":
    logger.info("Testing Recorder")
    q = Queue()
    r = Recorder(q)
    r.start_recording()
    input("Press Enter to stop recording")
    r.stop_recording()
    logger.info(f"Queue size: {q.qsize()}")