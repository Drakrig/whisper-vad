from queue import Queue
import sounddevice as sd
import numpy as np
from threading import Thread
from logging import getLogger
from log_config import setup_logging

setup_logging()
logger = getLogger(__name__)

class Recorder():
    def __init__(self,
                 output_queue:Queue,
                 sample_rate=16000,
                 frame_duration_ms=32):
        self.output_queue = output_queue
        self.sr = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.running = False

    def read_from_stream(self, indata, frames, time, status):
        if status:
            logger.error(f"Error: {status}")
        self.output_queue.put(np.array(indata, dtype=np.float32)[:,0]) 

if __name__ == "__main__":
    logger.info("Testing Recorder")
    q = Queue()
    r = Recorder(q)
    r.start_recording()
    input("Press Enter to stop recording")
    r.stop_recording()
    logger.info(f"Queue size: {q.qsize()}")