from .audio_utils import AudioUtils
from .logger import logger
from openai import OpenAI
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import tempfile
import queue
import time
import os


class WhisperAPI:
    def __init__(self, api_key: str, base_url: str, model: str = "whisper-large-v3"):
        """
        Initialize Whisper API client for transcription.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL of the API endpoint
            model: Whisper model to use for transcription
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip('/')
        )
        self.model = model
        logger.info(f"WhisperAPI initialized with model: {model}")
    
    def transcribe_audio_data(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio data using OpenAI-compatible API.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Transcribed text as string
        """
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file_path = tmp_file.name
            
            # Normalize and save audio
            audio_data = np.clip(audio_data, -1.0, 1.0)
            sf.write(tmp_file_path, audio_data, sample_rate, subtype='PCM_16')
            
            # Transcribe using OpenAI client
            with open(tmp_file_path, 'rb') as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language='ru',  # Force Russian language
                    response_format="text"
                )
            
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            
            text = str(transcription).strip()
            logger.info(f"API transcription: '{text}'")
            return text
            
        except Exception as e:
            logger.error(f"Error in API transcription: {e}")
            # Clean up temp file in case of error
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
            return ""
    
    def transcribe_file(self, file_path: str) -> str:
        """
        Transcribe audio file using API.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Transcribed text as string
        """
        try:
            with open(file_path, 'rb') as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language='ru',
                    response_format="text"
                )
            
            text = str(transcription).strip()
            logger.info(f"API file transcription: '{text}'")
            return text
            
        except Exception as e:
            logger.error(f"Error transcribing file via API: {e}")
            return ""


class StreamRecorderAPI:
    """
    Real-time audio stream recorder for API-based transcription.
    
    Records audio from microphone, buffers it, and provides chunks
    for transcription when speech is detected.
    """
    
    def __init__(self, api_transcriber, sample_rate=16000, chunk_duration=5.0, min_interval=2.0):
        """
        Initialize stream recorder.
        
        Args:
            api_transcriber: WhisperAPI instance for transcription
            sample_rate: Audio sampling rate in Hz
            chunk_duration: Duration of each audio chunk in seconds
            min_interval: Minimum time between API requests to avoid rate limiting
        """
        self.api_transcriber = api_transcriber
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.min_interval = min_interval  # Rate limiting for API calls
        
        # Stream control state
        self.is_recording = False
        self.audio_queue = queue.Queue()  # Thread-safe buffer for audio data
        self.device_id = None  # Audio input device ID
        self.audio_buffer = np.array([], dtype=np.float32)  # Accumulated audio samples
        self.last_transcription_time = 0  # Timer for rate limiting
        
        logger.info("StreamRecorderAPI initialized")
    
    def set_device(self, device_id):
        """Set audio input device for recording."""
        self.device_id = device_id
        logger.info(f"Audio device set to: {device_id}")
    
    def start_recording(self):
        """Start audio recording in background thread."""
        self.is_recording = True
        self.audio_buffer = np.array([], dtype=np.float32)  # Clear previous data
        self.last_transcription_time = 0  # Reset rate limiting timer
        
        # Start recording in separate thread to avoid blocking
        self.recording_thread = threading.Thread(target=self._record_audio, daemon=True)
        self.recording_thread.start()
        
        logger.info("API recording started")
    
    def stop_recording(self):
        """Stop audio recording and clean up resources."""
        self.is_recording = False
        logger.info("API recording stopped")
    
    def _record_audio(self):
        """
        Audio recording loop running in background thread.
        
        Continuously captures audio from the selected input device
        and adds it to the buffer queue.
        """
        def callback(indata, frames, time, status):
            """
            Callback function for sounddevice input stream.
            
            This runs in a high-priority audio thread, so it must
            be efficient and avoid blocking operations.
            """
            if status:
                logger.warning(f"Audio stream status: {status}")
                
            if self.is_recording:
                # Convert to mono if necessary and ensure correct data type
                audio_data = indata[:, 0] if indata.ndim > 1 else indata
                self.audio_queue.put(audio_data.astype(np.float32))
        
        try:
            # Configure and start audio input stream
            with sd.InputStream(
                device=self.device_id,
                channels=1,  # Mono audio for speech recognition
                samplerate=self.sample_rate,
                callback=callback,
                blocksize=int(self.sample_rate * 0.1)  # 100ms blocks for low latency
            ):
                # Keep stream active while recording
                while self.is_recording:
                    sd.sleep(100)  # Non-busy wait to reduce CPU usage
                    
        except Exception as e:
            logger.error(f"Recording error: {e}")
    
    def get_audio_chunk(self):
        """
        Extract an audio chunk from buffer if it contains speech.
        
        Implements:
        - Rate limiting to avoid excessive API calls
        - Speech detection to skip silent chunks
        - Buffer management for real-time performance
        
        Returns:
            Audio chunk as numpy array or None if no speech detected
        """
        frames_per_chunk = int(self.sample_rate * self.chunk_duration)
        
        # Collect available audio data from queue
        while not self.audio_queue.empty() and len(self.audio_buffer) < frames_per_chunk * 2:
            try:
                chunk = self.audio_queue.get_nowait()
                self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
            except queue.Empty:
                break
        
        # Apply rate limiting to avoid overwhelming the API
        current_time = time.time()
        if current_time - self.last_transcription_time < self.min_interval:
            return None
        
        # Check if we have enough audio for a complete chunk
        if len(self.audio_buffer) >= frames_per_chunk:
            audio_chunk = self.audio_buffer[:frames_per_chunk]
            
            # Only process chunks that contain actual speech
            if AudioUtils.has_speech(audio_chunk):
                # Remove processed audio from buffer
                self.audio_buffer = self.audio_buffer[frames_per_chunk:]
                self.last_transcription_time = current_time
                return audio_chunk
            else:
                # Discard silent chunks to prevent buffer overflow
                self.audio_buffer = self.audio_buffer[frames_per_chunk:]
                return None
        
        return None
    
    def process_chunk(self):
        """
        Process the next available audio chunk through API transcription.
        
        Returns:
            Transcribed text if speech was detected and processed, None otherwise
        """
        audio_chunk = self.get_audio_chunk()
        if audio_chunk is not None and len(audio_chunk) > 0:
            try:
                # Send audio chunk to API for transcription
                text = self.api_transcriber.transcribe_audio_data(audio_chunk, self.sample_rate)
                return text
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                return None
        return None