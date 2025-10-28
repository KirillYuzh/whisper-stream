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