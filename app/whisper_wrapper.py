from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import numpy as np
from threading import Thread
import logging
from logger_setup import setup_custom_logger

logger = setup_custom_logger(__name__, 'logs/whisper.log', log_level=logging.INFO) 

class S2T():
    def __init__(self, 
                 model_path:str,
                 input_queue:Queue,
                 output_queue:Queue):
        logger.info(f"Loading model from {model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained("app/model/whisper")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "app/model/whisper", 
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
        
        self.input_queue = input_queue
        self.output_queue = output_queue
        
        logger.info("Pipeline ready")

        self.running = False
    
    def __call__(self, data:np.ndarray):
        logger.debug(f"Transcribing audio chunk of size {data.shape}")
        text = self.pipe(data, generate_kwargs={"language": "en"})["text"]
        logger.debug(f"Transcription: {text}")
        return text
    
    def process_audio(self):
        while self.running:
            data = self.input_queue.get()
            text = self(data)
            self.output_queue.put(text)
    
    def start_whisper(self):
        self.running = True
        self.thread = Thread(target=self.process_audio)
        self.thread.start()
        
    def stop_whisper(self):
        self.running = False
        self.thread.join()

if __name__ == "__main__":
    logger.info("Testing Whisper")
    in_q, out_q = Queue(), Queue()
    s2t = S2T("app/model/whisper", in_q, out_q)
    s2t.start_whisper()
    input("Press Enter to stop whispering")
    s2t.stop_whisper()