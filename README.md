# Whisper-tools
High-level python library for stream and static transcription with whisper

# Getting started

Installation 
```bash
pip install whisper-tools
```

# Transcribing an audio file

Transcribing an audio file locally:
```python
from whisper_tools import WhisperLocal

text = WhisperLocal().transcribe_file("/voice example/example.wav")

print(text)
```

Transcribing an audio file via API:
```python
from whisper_tools import WhisperAPI

whisper_api = WhisperAPI(api_key="your_key", base_url="your_url")
text = whisper_api.transcribe_file_api("/voice example/example.wav")

print(text)
```

# Real-time transcription

> [!IMPORTANT]  
> True streaming transcription requires modifications to the Whisper architecture, as the original model expects a complete audio file, so we send information in chunks.

Streaming transcription locally:
```python
from whisper_tools import WhisperLocal, StreamRecorder

recorder = StreamRecorder(WhisperLocal())

try:
    # start recording
    recorder.start_recording()
    print("Recording... Press Ctrl+C to stop")
    while True:
        # get a chunk (block of transcribed speech)
        text = recorder.process_chunk()
        if text:
            print(text)
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    # stop recording
    recorder.stop_recording()
```

Or we can write to the file:
```python
try:
    f = open('transcribed.txt', 'w')
    # start recording
    recorder.start_recording()
    print("Recording... Press Ctrl+C to stop")
    while True:
        # get a chunk (block of transcribed speech)
        text = recorder.process_chunk()
        if text:
            f.write(text + '\n')
            f.flush() 
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    f.close()
    # stop recording
    recorder.stop_recording()
```

Streaming transcription via API:
```python
from whisper_tools import WhisperAPI, StreamRecorderAPI

recorder = StreamRecorderAPI(WhisperAPI(api_key="your_key", base_url="your_url"))

try:
    recorder.start_recording()
    print("Recording... Press Ctrl+C to stop")
    while True:
        text = recorder.process_chunk()
        if text:
            print(text)
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    recorder.stop_recording()
```


