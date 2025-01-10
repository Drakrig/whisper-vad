"""Module to define the tasks to be executed
The general idea is to have a Task class that can be inherited by other classes.
Each subclass should redefine the run method to define the task to be executed."""

from multiprocessing import Condition, Event, JoinableQueue
from recorder import Recorder
from vad import VAD
from whisper import WhisperWrapper
from dataclasses import dataclass
import numpy as np
from queue import Empty
from typing import Optional
from logging import getLogger
from log_config import setup_logging

setup_logging()
logger = getLogger(__name__)

@dataclass(kw_only=True)
class Task:
    '''Basic class blueprint for a task
    Expected to be inherited by other classes with redifined run method.
    Expected behaviour is to run the task in a loop until stop_event is set.
    Attributes:
    :param name: name of the task
    :type name: str
    :param stop_event: Event object to stop the task
    :type stop_event: :class:`Event`
    :param run_condition: Condition object for synchronization, default is None
    :type run_condition: :class:`Condition`
    :param output_queue: Queue to put the output of the task
    :type output_queue: :class:`JoinableQueue'
    :param input_queue: Queue to get the input of the task, default is None
    :type input_queue: :class:`JoinableQueue'
    :param next_task_notifier: Condition object to notify the next task, default is None
    :type next_task_notifier: :class:`Condition'
    '''
    name: str
    output_queue: JoinableQueue
    stop_event: Event
    run_condition: Optional[Condition] = None
    input_queue: Optional[JoinableQueue] = None
    next_task_notifier: Optional[Condition] = None

    def run(self):
        '''Method to be redefined by subclasses'''
        pass

@dataclass(kw_only=True)
class RecorderTask(Task):
    recorder: Recorder

    def run(self):
        self.recorder.start_recording()
        try:
            while not self.stop_event.is_set():
                data = recorder.audio_stream.read(self.frame_size)
                data = np.frombuffer(data, dtype=np.float32)
                self.output_queue.put(data)
        except KeyboardInterrupt:
            logger.info(f"Task {self.name} stopped")

@dataclass(kw_only=True)
class VADTask(Task):
    vad: VAD
    max_silence_duration_ms: int = 1000
    frame_duration_ms: int = 32
    threshold: float = 0.5

    def run(self, *args, **kwargs):
        buffer = []
        silence_chunk_counter = 0
        chunk_limmit = self.max_silence_duration_ms // self.frame_duration_ms
        logger.debug(f"Silence chunk limit: {chunk_limmit}")
        try:
            while not self.stop_event.is_set():
                try:
                    data = self.input_queue.get(timeout=1)
                except Empty:
                    continue
                self.input_queue.task_done()
                prob = self.vad(data, self.vad.sr)[0][0]
                logger.debug(f"VAD output: {prob}")
                if prob > self.threshold:
                    buffer.append(data)
                    if silence_chunk_counter:
                        silence_chunk_counter = 0
                else:
                    silence_chunk_counter+=1
                    logger.debug(f"Silence chunk counter: {silence_chunk_counter}")
                    if silence_chunk_counter > chunk_limmit and len(buffer) > 0:
                        logger.debug("Silence limit reached. Processing audio")
                        self.output_queue.put(np.concatenate(buffer))
                        buffer = []
                        silence_chunk_counter = 0
                        with self.next_task_notifier:
                            self.next_task_notifier.notify()
        except KeyboardInterrupt:
            logger.info(f"Task {self.name} stopped")

@dataclass(kw_only=True)
class WhisperTask(Task):
    whisper: WhisperWrapper

    def run(self, *args, **kwargs):
        try:
            while not self.stop_event.is_set():
                with self.run_condition:
                    try:
                        self.run_condition.wait(timeout=1)
                    except RuntimeError:
                        continue
                try:
                    data = self.input_queue.get(timeout=1)
                except Empty:
                    logger.error(f"Timeout waiting for input in {self.name}")
                    continue
                self.input_queue.task_done()
                text = self.whisper(data)
                self.output_queue.put(text)
        except KeyboardInterrupt:
            logger.info(f"Task {self.name} stopped")