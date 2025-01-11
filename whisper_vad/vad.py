import onnxruntime
import numpy as np
from logging import getLogger
from log_config import setup_logging

setup_logging()
logger = getLogger(__name__)

class VAD():
    """Voice Activity Detection class
    Original code from https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py
    """
    def __init__(self, 
                 model_path,
                 device="cpu"):
        logger.info(f"Loading VAD model from {model_path}")
        self.device = device
        self.model_path = model_path
        self.sr = 16000
        self._reset_states()
        logger.info("VAD model loaded")
    
    def prepare_session(self):
        """Make necessary preparations for the ONNX Runtime inference
        """
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        if self.device == "cpu":
            self.session = onnxruntime.InferenceSession(self.model_path, providers=["CPUExecutionProvider"],sess_options=opts)
        else:
            self.session = onnxruntime.InferenceSession(self.model_path, sess_options=opts)
    
    def _validate_input(self, x:np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """Check input audio chunk for mismatch size or shape

        :param x: Input audio chunk
        :type x: np.ndarray
        :param sr: Sample rate
        :type sr: int
        :raises ValueError: When input audio chunk has too many dimensions
        :raises ValueError: When input audio chunk is too short
        :return: Tuple of input audio chunk (reshaped if nessesary) and sample rate
        :rtype: tuple[np.ndarray, int]
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        if len(x.shape) > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {len(x.shape)}")

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[:,::step]
            sr = 16000
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x, sr
    
    def _reset_states(self, batch_size=1):
        """Reset the states of the model"""
        self._state = np.zeros((2, batch_size, 128),dtype=np.float32)
        self._context = np.zeros(0,dtype=np.float32)
        self._last_sr = 0
        self._last_batch_size = 0
    
    def __call__(self, x:np.ndarray, sr:int) -> list[list[float]]:
        """Main model inference function

        :param x: INput audio chunk
        :type x: np.ndarray
        :param sr: Sample rate
        :type sr: int
        :raises ValueError: When mismatch expected number of samples
        :raises ValueError: When sample rate is not supported
        :return: Raw output from inference session as list of voice activity probabilities
        :rtype: list[list[float]]
        """
        x, sr = self._validate_input(x, sr)
        num_samples = 512 if sr == 16000 else 256

        if x.shape[-1] != num_samples:
            raise ValueError(f"Provided number of samples is {x.shape[-1]} (Supported values: 256 for 8000 sample rate, 512 for 16000)")

        batch_size = x.shape[0]
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size:
            self._reset_states(batch_size)
        if (self._last_sr) and (self._last_sr != sr):
            self._reset_states(batch_size)
        if (self._last_batch_size) and (self._last_batch_size != batch_size):
            self._reset_states(batch_size)

        if not len(self._context):
            self._context = np.zeros((batch_size, context_size), dtype=np.float32)

        x = np.concatenate([self._context, x], axis=1)
        if sr in [8000, 16000]:
            ort_inputs = {'input': x, 'state': self._state, 'sr': np.array(sr, dtype='int64')}
            ort_outs = self.session.run(None, ort_inputs)
            out, self._state = ort_outs
        else:
            raise ValueError()

        self._context = x[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size

        return out

if __name__ == "__main__":
    logger.info("Testing VAD")
    vad = VAD("model/vad/model.onnx")
    x = np.random.randn(1, 512)
    out = vad(x, 16000)
    logger.info(out)