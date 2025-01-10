from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import numpy as np

class WhisperWrapper():
    def __init__(self, 
                 model_path:str):
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
        self.thread.join()
        logger.info("Whisper stopped")