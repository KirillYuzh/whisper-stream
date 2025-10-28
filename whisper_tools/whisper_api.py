from .audio_utils import AudioUtils
from .logger import logger
from openai import OpenAI
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import requests
import tempfile
import queue
import time
import os


class WhisperAPI:
    def __init__(self, api_key: str, base_url: str, model: str = "whisper-large"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def transcribe_audio_data(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """transcribe audio via API"""
        try:
            # saving audio to the temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file_path = tmp_file.name
            
            # normalize audio
            audio_data = np.clip(audio_data, -1.0, 1.0)
            sf.write(tmp_file_path, audio_data, sample_rate, subtype='PCM_16')
            
            # sending to the API
            with open(tmp_file_path, 'rb') as audio_file:
                files = {
                    'file': ('audio.wav', audio_file, 'audio/wav'),
                    'model': (None, self.model),
                    'language': (None, 'ru'),
                    'response_format': (None, 'json')
                }
                
                response = self.session.post(
                    f"{self.base_url}",
                    files=files
                )
            
            # removing temp file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('text', '').strip()
                logger.info(f"API transcription: '{text}'")
                return text
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error in API transcription: {e}")
            # removing tmp file
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
            return ""
    
    def transcribe_file(self, file_path: str) -> str:
        """transcribe file via API"""
        try:
            with open(file_path, 'rb') as audio_file:
                files = {
                    'file': ('audio.wav', audio_file, 'audio/wav'),
                    'model': (None, self.model),
                    'language': (None, 'ru'),
                    'response_format': (None, 'json')
                }
                
                response = self.session.post(
                    f"{self.base_url}",
                    files=files
                )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('text', '').strip()
                logger.info(f"API file transcription: '{text}'")
                return text
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error transcribing file via API: {e}")
            return ""
        
class StreamRecorderAPI:
    def __init__(self, api_transcriber, sample_rate=16000, chunk_duration=5, min_interval=2.0):
        self.api_transcriber = api_transcriber
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.min_interval = min_interval
        
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.device_id = None
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_transcription_time = 0
        
        logger.info("StreamRecorderAPI initialized")
    
    def set_device(self, device_id):
        """set listening device"""
        self.device_id = device_id
        logger.info(f"Audio device set to: {device_id}")
    
    def start_recording(self):
        """start recording"""
        self.is_recording = True
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_transcription_time = 0
        
        self.recording_thread = threading.Thread(target=self._record_audio, daemon=True)
        self.recording_thread.start()
        
        logger.info("API recording started")
    
    def stop_recording(self):
        """stop recording"""
        self.is_recording = False
        logger.info("API recording stopped")
    
    def _record_audio(self):
        """audio flow"""
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
        """get chunk for translation (in case it contains speech)"""
        frames_per_chunk = int(self.sample_rate * self.chunk_duration)
        
        # collect data from the queue
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
        
        # if collected enough data
        if len(self.audio_buffer) >= frames_per_chunk:
            audio_chunk = self.audio_buffer[:frames_per_chunk]
            
            # check speech in chunk
            if AudioUtils.has_speech(audio_chunk):
                self.audio_buffer = self.audio_buffer[frames_per_chunk:]
                self.last_transcription_time = current_time
                return audio_chunk
            else:
                # skip silence chunk
                self.audio_buffer = self.audio_buffer[frames_per_chunk:]
                return None
        
        return None
    
    def process_chunk(self):
        """go to the next chunk via API"""
        audio_chunk = self.get_audio_chunk()
        if audio_chunk is not None and len(audio_chunk) > 0:
            text = self.api_transcriber.transcribe_audio_data(audio_chunk, self.sample_rate)
            return text
        return None