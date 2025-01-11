from multiprocessing import Process, Event, Condition, JoinableQueue, Manager
from threading import Thread
from queue import Empty
from recorder import Recorder
from vad import VAD
from whisper import WhisperWrapper
from multiproc import RecorderTask, VADTask, WhisperTask 
from logging import getLogger
from log_config import setup_logging
from torch.multiprocessing.spawn import spawn

if __name__ == "__main__":
    setup_logging()
    logger = getLogger(__name__)
    with Manager() as manager:
        recorder_input_queue = JoinableQueue()
        recorder_output_queue = JoinableQueue()
        vad_output_queue = JoinableQueue()
        whisper_output_queue = JoinableQueue()
        stop_event = Event()
        whisper_run_condition = Condition()

        whisper = WhisperWrapper("model/whisper/")

        vad = VAD("model/vad/model.onnx")

        recorder = Recorder(recorder_output_queue)

        whisper_task = WhisperTask(
            name="Whisper",
            output_queue=whisper_output_queue,
            stop_event=stop_event,
            input_queue=vad_output_queue,
            whisper=whisper,
            run_condition=whisper_run_condition)
        whisper_process = Thread(name="Whisper", target=whisper_task.run)

        vad_task = VADTask(
            name="VAD", 
            output_queue=vad_output_queue, 
            stop_event=stop_event, 
            input_queue=recorder_output_queue, 
            next_task_notifier=whisper_run_condition,
            vad=vad)
        vad_process = Process(name="VAD", target=vad_task.run)

        recorder_task = RecorderTask(
            name="Recorder",
            input_queue=recorder_input_queue,
            output_queue=recorder_output_queue,
            stop_event=stop_event,
            recorder=recorder)
        recorder_process = Process(name="Recorder", target=recorder_task.run)

        try:
            whisper_process.start()
            vad_process.start()
            recorder_process.start()
            logger.info("All tasks started")
            while True:
                try:
                    text = whisper_output_queue.get(timeout=1)
                except Empty:
                    continue
                whisper_output_queue.task_done()
                logger.info(f"Transcription: {text}")
        except KeyboardInterrupt:
            logger.info("Stopping tasks")
            stop_event.set()
            for process in [recorder_process, vad_process, whisper_process]:
                process.join(timeout=5)
                if process.is_alive():
                    logger.debug(f"Process {process.name} did not stop. Terminating...")
                    process.terminate()
            logger.info("All tasks stopped")
            logger.debug(f"Recorder queue size: {recorder_output_queue.qsize()}")
            logger.debug(f"VAD queue size: {vad_output_queue.qsize()}")
            logger.debug(f"S2T queue size: {whisper_output_queue.qsize()}")