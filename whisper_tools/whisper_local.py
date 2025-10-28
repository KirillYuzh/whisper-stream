from .audio_utils import AudioUtils
from .logger import logger
import platform
import torch

import sounddevice as sd
import numpy as np
import threading
import queue
import time

from faster_whisper import WhisperModel


class WhisperLocal:
    def __init__(self, model_name="base", language='ru', device='auto'):
        """
        Initialize Whisper transcription service using faster-whisper.
        
        faster-whisper is a reimplementation of Whisper that's significantly faster
        and more memory-efficient than the original Hugging Face implementation.
        
        Args:
            model_name: Model size ("tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3")
            language: Target language for transcription
            device: Hardware device to run on ('auto', 'cpu', 'cuda', etc.)
        """
        self.model_name = model_name
        self.language = language
        self.device = device
        self.model = None  # faster-whisper model instance
        self._load_model()

    def _get_device(self):
        """
        Determine the optimal computation device for model inference.
        
        faster-whisper supports CPU and GPU execution with different compute types
        for balancing speed and memory usage.
        """
        if self.device != 'auto':
            return self.device
        
        # Check for NVIDIA GPU with CUDA
        if torch.cuda.is_available():
            logger.info("Using CUDA device")
            return "cuda"
        
        # Fall back to CPU if no GPU available
        logger.info("Using CPU device")
        return "cpu"
    
    def _get_compute_type(self, device):
        """
        Determine the appropriate compute type for the selected device.
        
        Compute types control the precision of calculations:
        - float16: Faster but less precise (good for GPUs)
        - int8: Even faster with quantization (good for CPU/limited memory)
        - float32: Most precise but slowest
        """
        if device == "cuda":
            # Use half-precision for GPU acceleration
            return "float16"
        elif device == "mps":
            # MPS currently has limited support in faster-whisper
            return "float16"
        else:
            # Use 8-bit integer quantization for CPU efficiency
            return "int8"
    
    def _load_model(self):
        """Load the faster-whisper model with optimized settings."""
        try:
            logger.info(f"Loading faster-whisper model: {self.model_name}")
            
            device = self._get_device()
            compute_type = self._get_compute_type(device)
            
            # Initialize faster-whisper model
            # device_index allows multi-GPU support, we use single device
            # compute_type balances speed vs precision
            self.model = WhisperModel(
                model_size_or_path=self.model_name,
                device=device,
                compute_type=compute_type,
                device_index=0,  # Use first available device
                num_workers=1,   # Single worker for thread safety
            )
            logger.info(f"faster-whisper model loaded successfully on {device} with {compute_type} precision")
        except Exception as e:
            logger.error(f"Error loading faster-whisper model: {e}")
            raise
    
    def _transcribe_chunk(self, chunk_data, sample_rate):
        """
        Transcribe a single audio chunk using faster-whisper.
        
        faster-whisper processes audio directly from numpy arrays or file paths
        and returns segments with timing information.
        """
        try:
            # Convert numpy array to the format expected by faster-whisper
            # The model expects mono audio at 16kHz
            audio_array = chunk_data.astype(np.float32)
            
            # Transcribe using faster-whisper's segment-based approach
            # beam_size=1 uses greedy decoding for speed, language forces target language
            segments, info = self.model.transcribe(
                audio=audio_array,
                language=self.language,
                task="transcribe",
                beam_size=1,  # Faster decoding with minimal quality loss
                best_of=1,    # Single candidate generation for speed
                temperature=0.0,  # Deterministic output
                vad_filter=True,  # Voice activity detection to filter non-speech
                vad_parameters=dict(min_silence_duration_ms=500),  # Aggressive VAD
            )
            
            # Collect text from all segments
            segment_texts = []
            for segment in segments:
                if segment.text.strip():
                    segment_texts.append(segment.text.strip())
            
            # Combine segments into final text
            text = " ".join(segment_texts)
            logger.info(f"Chunk result: '{text}'")
            return text
            
        except Exception as e:
            logger.error(f"Error transcribing chunk: {e}")
            return ""
        
    def _split_audio_into_chunks(self, audio_data, sample_rate):
        """
        Split audio into overlapping chunks for sequential processing.
        
        Chunking helps with:
        - Processing long audio files that don't fit in memory
        - Providing more frequent updates during streaming
        - Better handling of context changes in long recordings
        """
        # 10-second chunks provide good context while being memory efficient
        chunk_size_samples = 10 * sample_rate  
        # 2-second overlap prevents word cutting at chunk boundaries
        overlap_samples = 2 * sample_rate      
        # Effective step size after accounting for overlap
        step_samples = chunk_size_samples - overlap_samples
        
        chunks = []
        start = 0
        
        # Create overlapping chunks across the entire audio
        while start < len(audio_data):
            end = min(start + chunk_size_samples, len(audio_data))
            chunk = audio_data[start:end]
            
            # Only include chunks longer than 2 seconds to avoid very short segments
            if len(chunk) > sample_rate * 2:
                chunks.append(chunk)
            
            # Stop if we've reached the end
            if end >= len(audio_data):
                break
                
            # Move to next chunk with overlap
            start += step_samples
        
        logger.info(f"Split audio into {len(chunks)} chunks")
        return chunks
   
    def transcribe_file(self, file_path, sample_rate=16000):
        """
        Transcribe an audio file by splitting it into manageable chunks.
        
        This approach is essential for:
        - Very long audio files that exceed memory limits
        - Real-time processing where incremental results are needed
        - Better error recovery (failed chunk doesn't break entire process)
        """
        try:
            logger.info(f"Transcribing file with chunking: {file_path}")

            # Read and preprocess audio file
            result = AudioUtils.read_audio_file(file_path)
            if result is None:
                logger.error("Failed to read audio file")
                return ""
            
            sample_rate, audio_data = result
            
            # Split audio into sequential chunks with overlap
            chunks = self._split_audio_into_chunks(audio_data, sample_rate)
            
            if not chunks:
                logger.warning("No chunks created from audio")
                return ""
            
            # Process each chunk sequentially
            all_texts = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Transcribing chunk {i+1}/{len(chunks)}")
                text = self._transcribe_chunk(chunk, sample_rate)
                if text and text.strip():
                    all_texts.append(text.strip())
            
            # Combine all chunk results into final transcription
            final_text = " ".join(all_texts)
            logger.info(f"Final transcription: {final_text}")
                
            return final_text
            
        except Exception as e:
            logger.error(f"Error transcribing with chunking: {e}")
            return ""

    def transcribe_audio_data(self, audio_data, sample_rate=16000):
        """
        Transcribe raw audio data by saving to temporary file and processing.
        
        This method provides flexibility to handle both file paths and raw audio data
        with the same processing pipeline.
        """
        try:
            # Save audio data to temporary file for processing
            tmp_file_path = AudioUtils.save_audio_to_temp(audio_data, sample_rate)
            if not tmp_file_path:
                return ""
            
            # Use file-based transcription for consistency
            text = self.transcribe_file(tmp_file_path)
            
            # Clean up temporary file to avoid disk space accumulation
            AudioUtils.cleanup_temp_file(tmp_file_path)
            return text
            
        except Exception as e:
            logger.error(f"Error transcribing audio data: {e}")
            return ""
        

class StreamRecorder:
    """
    Real-time audio stream recorder and processor.
    
    This class handles continuous audio input from microphones or other sources,
    buffers the data, and provides chunks ready for transcription when speech
    is detected.
    """
    
    def __init__(self, transcriber, sample_rate=16000, chunk_duration=2.1, min_interval=1.0):
        """
        Initialize stream recorder for real-time audio processing.
        
        Args:
            transcriber: WhisperLocal instance for transcription
            sample_rate: Audio sampling rate (Hz)
            chunk_duration: Length of each audio chunk in seconds
            min_interval: Minimum time between transcription attempts
        """
        self.transcriber = transcriber
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.min_interval = min_interval  # Prevents excessive API calls
        
        # Stream control flags
        self.is_recording = False
        self.audio_queue = queue.Queue()  # Thread-safe audio data buffer
        self.device_id = None  # Audio input device identifier
        self.audio_buffer = np.array([], dtype=np.float32)  # Accumulated audio samples
        self.last_transcription_time = 0  # Rate limiting timer
        
        logger.info("StreamRecorder initialized")
    
    def set_device(self, device_id):
        """Set the audio input device for recording."""
        # Validate device exists before setting
        devices = AudioUtils.list_input_devices()
        self.device_id = devices[0]['index']  # Use first available device by default
        logger.info(f"Audio device set to: {device_id}")
    
    def start_recording(self):
        """Begin audio recording in a separate thread."""
        self.is_recording = True
        self.audio_buffer = np.array([], dtype=np.float32)  # Clear previous data
        self.last_transcription_time = 0  # Reset timing
        
        # Start recording in background thread to avoid blocking main program
        self.recording_thread = threading.Thread(target=self._record_audio, daemon=True)
        self.recording_thread.start()
        
        logger.info("Recording started")
    
    def stop_recording(self):
        """Stop audio recording and clean up resources."""
        self.is_recording = False
        logger.info("Recording stopped")
    
    def _record_audio(self):
        """
        Audio recording loop running in separate thread.
        
        Uses sounddevice's callback system to continuously capture audio
        from the selected input device and buffer it for processing.
        """
        def callback(indata, frames, time, status):
            """
            Callback function called by sounddevice for each audio block.
            
            This runs in a high-priority audio thread, so it must be efficient
            and avoid blocking operations.
            """
            if status:
                logger.warning(f"Audio stream status: {status}")
                
            if self.is_recording:
                # Convert multi-channel to mono if necessary and ensure correct dtype
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
                # Keep the stream active while recording
                while self.is_recording:
                    sd.sleep(100)  # Non-busy wait to reduce CPU usage
                    
        except Exception as e:
            logger.error(f"Recording error: {e}")
    
    def get_audio_chunk(self):
        """
        Extract a chunk of audio from buffer if it contains speech.
        
        Implements several important features:
        - Rate limiting to prevent excessive transcription requests
        - Speech detection to avoid processing silence
        - Buffer management to maintain real-time performance
        """
        # Calculate required samples for one chunk
        frames_per_chunk = int(self.sample_rate * self.chunk_duration)
        
        # Collect available audio data from queue without blocking
        while not self.audio_queue.empty() and len(self.audio_buffer) < frames_per_chunk * 2:
            try:
                chunk = self.audio_queue.get_nowait()
                self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
            except queue.Empty:
                break
        
        # Enforce minimum time between transcriptions to avoid overwhelming the system
        current_time = time.time()
        if current_time - self.last_transcription_time < self.min_interval:
            return None
        
        # Check if we have enough audio for a complete chunk
        if len(self.audio_buffer) >= frames_per_chunk:
            audio_chunk = self.audio_buffer[:frames_per_chunk]
            
            # Only return chunks that contain actual speech
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
        Process the next available audio chunk through the transcription pipeline.
        
        Returns:
            Transcribed text if speech was detected and processed, None otherwise
        """
        audio_chunk = self.get_audio_chunk()
        if audio_chunk is not None and len(audio_chunk) > 0:
            # Send audio chunk to transcription service
            text = self.transcriber.transcribe_audio_data(audio_chunk, self.sample_rate)
            return text
        return None