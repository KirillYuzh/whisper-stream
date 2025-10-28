from .logger import logger
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os


class AudioUtils:
    @staticmethod
    def list_input_devices():
        """list of available audio devices"""
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                })
        return input_devices
    
    @staticmethod
    def has_speech(audio_chunk, energy_threshold=0.00001, dynamic_threshold=0.005):
        """check speech in chunk"""
        if len(audio_chunk) == 0:
            return False
        
        energy = np.mean(audio_chunk ** 2)
        dynamic_range = np.max(audio_chunk) - np.min(audio_chunk)
        
        has_speech = energy > energy_threshold and dynamic_range > dynamic_threshold
        return has_speech
    
    @staticmethod
    def save_audio_to_temp(audio_data, sample_rate=16000):
        """save audio to the temp file"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file_path = tmp_file.name
            
            audio_data = np.array(audio_data)
            
            # if it's multidimensional we take the first channel
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0] if audio_data.shape[1] > 1 else audio_data.flatten()
            else:
                audio_data = audio_data.flatten()
                
            sf.write(tmp_file_path, audio_data, sample_rate, subtype='PCM_16')
            logger.debug(f"Audio saved to temp file: {tmp_file_path}")
            return tmp_file_path
        except Exception as e:
            logger.error(f"Error saving audio to temp file: {e}")
            return None
    
    @staticmethod
    def cleanup_temp_file(file_path):
        """remove temp file"""
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except Exception as e:
                logger.warning(f"Error cleaning up temp file: {e}")

    @staticmethod
    def read_audio_file(file_path):
        """read audio file"""
        try:
            audio_data, sample_rate = sf.read(file_path, dtype='float32')
            
            # if it's multidimensional we take the first channel
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0] if audio_data.shape[1] > 1 else audio_data.flatten()
            else:
                audio_data = audio_data.flatten()
                
            return sample_rate, audio_data
        except Exception as e:
            logger.error(f"Error reading audio file with soundfile: {e}")