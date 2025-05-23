from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import numpy as np
from logging import getLogger
from log_config import setup_logging

setup_logging()
logger = getLogger(__name__)

class WhisperWrapper():
    """Wrapper for the HuggingFace pipeline for ASR
    """
    def __init__(self, 
                 model_path:str):
        logger.info(f"Loading model from {model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
            )
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device)
    
        
        logger.info("Pipeline ready")

        self.running = False
    
    def __call__(self, data:np.ndarray) -> str:
        """Process audio data

        :param data: Audio data
        :type data: np.ndarray
        :return: Recognized text
        :rtype: str
        """
        logger.debug(f"Transcribing audio chunk of size {data.shape}")
        text = self.pipe(data, generate_kwargs={"language": "en"})["text"]
        logger.debug(f"Transcription: {text}")
        return text