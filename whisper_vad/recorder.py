from queue import Queue
import sounddevice as sd
import numpy as np
from threading import Thread
from logging import getLogger
from log_config import setup_logging

setup_logging()
logger = getLogger(__name__)

class Recorder():
    """Class to provide recording functionality

    :param output_queue: Queue to store recorded audio chunks. Might be any instance of Queue that support `put` method like `multiprocessing.Queue`
    :type output_queue: Queue
    :param sample_rate: Recording sample rate, defaults to 16000
    :type sample_rate: int, optional
    :param frame_duration_ms: Duration of one audio chunk, supported by VAD model, defaults to 32
    :type frame_duration_ms: int, optional
    """
    def __init__(self,
                 output_queue:Queue,
                 sample_rate=16000,
                 frame_duration_ms=32):
        self.output_queue = output_queue
        self.sr = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.running = False

    def read_from_stream(self, 
                         indata: np.ndarray,
                         frames: int, 
                         time, 
                         status):
        """Function compatible with `sounddevice.InputStream.read` method

        :param indata: Buffer with recorded audio data
        :type indata: _type_
        :param frames: Amount of frames in the buffer
        :type frames: int
        :param time: CFFI structure with timestamps
        :type time: CData
        :param status: Indicating whether input and/or output buffers have been inserted or will be dropped to overcome underflow or overflow conditions.
        :type status: CallbackFlags
        """
        if status:
            logger.error(f"Error: {status}")
        self.output_queue.put(indata.astype(np.float32)[:,0]) 

if __name__ == "__main__":
    logger.info("Testing Recorder")
    q = Queue()
    r = Recorder(q)
    r.start_recording()
    input("Press Enter to stop recording")
    r.stop_recording()
    logger.info(f"Queue size: {q.qsize()}")