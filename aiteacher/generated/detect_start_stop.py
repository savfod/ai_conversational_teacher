import vosk
import json
import sounddevice as sd
import numpy as np
import queue

class VoiceCommandDetector:
    def __init__(self, model_name: str):
        vosk.SetLogLevel(-1)
        self.model = vosk.Model(model_name)
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        self.command_queue = queue.Queue()
        self.commands = {
            "start": False,
            "end": False
        }

    def detect_commands(self, audio_chunk):
        if self.recognizer.AcceptWaveform(audio_chunk):
            result = json.loads(self.recognizer.Result())
            text = result.get('text', '').strip().lower()
            
            if "start" in text:
                self.commands["start"] = True
            if "stop" in text:
                self.commands["end"] = True
            
            return text
        return None

    def listen_continuously(self):
        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            
            # Convert to mono if stereo
            if indata.ndim > 1:
                indata = indata[:, 0]
            
            # Convert to 16-bit PCM
            audio_data = (indata * 32767).astype(np.int16).tobytes()
            
            # Detect commands
            detected_text = self.detect_commands(audio_data)
            if detected_text:
                print(f"Detected: {detected_text}")

        # Start listening
        with sd.InputStream(callback=audio_callback, 
                            channels=1, 
                            samplerate=16000, 
                            dtype='float32'):
            while True:
                if self.commands["start"]:
                    print("Start command detected!")
                    # Your start logic here
                    self.commands["start"] = False
                
                if self.commands["end"]:
                    print("End command detected!")
                    # Your end logic here
                    self.commands["end"] = False

if __name__ == "__main__":
    detector = VoiceCommandDetector("vosk-model-small-en-us-0.15")
    detector.listen_continuously()