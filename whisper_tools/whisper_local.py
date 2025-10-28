from .audio_utils import AudioUtils
from .logger import logger
import transformers 
import platform
import torch

import sounddevice as sd
import numpy as np
import threading
import queue
import time


# disable whisper log data
transformers.logging.set_verbosity_error()

class WhisperLocal:
    def __init__(self, model_name="openai/whisper-tiny", language='ru', device='auto'):
        self.model_name = model_name
        self.language = language
        self.device = device
        self.transcriber = None
        self._load_model()

    def _get_device(self):
        """detecting audio devices"""
        if self.device != 'auto':
            return self.device
            
        # checking platform
        # using gpu: mps (Apple Silicon) or cuda
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            logger.info("Using MPS (Apple Silicon) device")
            return "mps"
        
        if torch.cuda.is_available():
            logger.info("Using CUDA device")
            return "cuda"
        
        # using cpu if gpu is not available
        logger.info("Using CPU device")
        return "cpu"
    
    def _load_model(self):
        """load Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            
            device = self._get_device()
            
            self.transcriber = transformers.pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                language=self.language,
                device=device,
                return_timestamps=True,
            )
            logger.info(f"Whisper model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise
    
    def _transcribe_chunk(self, chunk_data, sample_rate):
        """translate one chunk"""
        try:
            tmp_file_path = AudioUtils.save_audio_to_temp(chunk_data, sample_rate)
            if not tmp_file_path:
                return ""
            
            result = self.transcriber(
                tmp_file_path,
                return_timestamps=True,  
                generate_kwargs={"language": self.language, "task": "transcribe"}
            )
            
            if isinstance(result, dict):
                text = result.get("text", "").strip()
            else:
                text = str(result).strip()
            
            AudioUtils.cleanup_temp_file(tmp_file_path)
            
            logger.info(f"Chunk result: '{text}'")
            return text
        except Exception as e:
            logger.error(f"Error transcribing chunk: {e}")
            return ""
        
    def _split_audio_into_chunks(self, audio_data, sample_rate):
        """splitting audio into chunks by 10 seconds and 2 seconds overlap"""
        chunk_size_samples = 10 * sample_rate  
        overlap_samples = 2 * sample_rate      
        step_samples = chunk_size_samples - overlap_samples
        
        chunks = []
        start = 0
        
        while start < len(audio_data):
            end = min(start + chunk_size_samples, len(audio_data))
            chunk = audio_data[start:end]
            
            # adding chunk if it is big enough 
            if len(chunk) > sample_rate * 2:  # 2 seconds
                chunks.append(chunk)
            
            if end >= len(audio_data):
                break
                
            start += step_samples
        
        logger.info(f"Split audio into {len(chunks)} chunks")
        return chunks
   
    def transcribe_file(self, file_path, sample_rate=16000):
        """translate audio file by chunks splitting"""
        try:
            logger.info(f"Transcribing file with chunking: {file_path}")

            result = AudioUtils.read_audio_file(file_path)
            if result is None:
                logger.error("Failed to read audio file")
                return ""
            
            sample_rate, audio_data = result
            
            # splitting - теперь передаем audio_data вместо file_path
            chunks = self._split_audio_into_chunks(audio_data, sample_rate)
            
            if not chunks:
                logger.warning("No chunks created from audio")
                return ""
            
            # translate chunk by chunk
            all_texts = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Transcribing chunk {i+1}/{len(chunks)}")
                text = self._transcribe_chunk(chunk, sample_rate)
                if text and text.strip():
                    all_texts.append(text.strip())
            
            # gluing the results
            final_text = " ".join(all_texts)
            logger.info(f"Final transcription: {final_text}")
                
            return final_text
            
        except Exception as e:
            logger.error(f"Error transcribing with chunking: {e}")
            return ""

    def transcribe_audio_data(self, audio_data, sample_rate=16000):
        """save audio to the temp file and translating it by calling transcribe_file()"""
        try:
            tmp_file_path = AudioUtils.save_audio_to_temp(audio_data, sample_rate)
            if not tmp_file_path:
                return ""
            
            text = self.transcribe_file(tmp_file_path)
            AudioUtils.cleanup_temp_file(tmp_file_path)
            return text
        except Exception as e:
            logger.error(f"Error transcribing audio data: {e}")
            return ""
        
class StreamRecorder:
    def __init__(self, transcriber, sample_rate=16000, chunk_duration=5.0, min_interval=2.0):
        self.transcriber = transcriber
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.min_interval = min_interval
        
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.device_id = None
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_transcription_time = 0
        
        logger.info("StreamRecorder initialized")
    
    def set_device(self, device_id):
        """set listening device"""
        devices = AudioUtils.list_input_devices()
        self.device_id = devices[0]['index']
        logger.info(f"Audio device set to: {device_id}")
    
    def start_recording(self):
        """start recording"""
        self.is_recording = True
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_transcription_time = 0
        
        self.recording_thread = threading.Thread(target=self._record_audio, daemon=True)
        self.recording_thread.start()
        
        logger.info("Recording started")
    
    def stop_recording(self):
        """stop recording"""
        self.is_recording = False
        logger.info("Recording stopped")
    
    def _record_audio(self):
        """recording flow"""
        def callback(indata, frames, time, status):
            if self.is_recording:
                audio_data = indata[:, 0] if indata.ndim > 1 else indata
                self.audio_queue.put(audio_data.astype(np.float32))
        
        try:
            with sd.InputStream(
                device=self.device_id,
                channels=1,
                samplerate=self.sample_rate,
                callback=callback,
                blocksize=int(self.sample_rate * 0.1)
            ):
                while self.is_recording:
                    sd.sleep(100)
        except Exception as e:
            logger.error(f"Recording error: {e}")
    
    def get_audio_chunk(self):
        """get cuhnks to translate (in case it contains speech)"""
        frames_per_chunk = int(self.sample_rate * self.chunk_duration)
        
        # collecting data from queue
        while not self.audio_queue.empty() and len(self.audio_buffer) < frames_per_chunk * 2:
            try:
                chunk = self.audio_queue.get_nowait()
                self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
            except queue.Empty:
                break
        
        # check interval between requests
        current_time = time.time()
        if current_time - self.last_transcription_time < self.min_interval:
            return None
        
        # case if we collected enough
        if len(self.audio_buffer) >= frames_per_chunk:
            audio_chunk = self.audio_buffer[:frames_per_chunk]
            
            # check speech in chunk
            if AudioUtils.has_speech(audio_chunk):
                self.audio_buffer = self.audio_buffer[frames_per_chunk:]
                self.last_transcription_time = current_time
                return audio_chunk
            else:
                # skip silent chunk
                self.audio_buffer = self.audio_buffer[frames_per_chunk:]
                return None
        
        return None
    
    def process_chunk(self):
        """translate next chunk"""
        audio_chunk = self.get_audio_chunk()
        if audio_chunk is not None and len(audio_chunk) > 0:
            text = self.transcriber.transcribe_audio_data(audio_chunk, self.sample_rate)
            return text
        return None