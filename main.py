import time
from aiteacher.audio.input_stream import MicrophoneInputStream, AudioFileInputStream
from aiteacher.generated.speech_api import speech_to_text
from aiteacher.generated.speech_api import text_to_speech
from aiteacher.audio.audio_parser import AudioParser
from aiteacher.generated.llm import answer

import sys

import sounddevice as sd

def main():
    """Main function to demonstrate MicrophoneInputStream usage."""
    print("Starting MicrophoneInputStream...")

    if sys.argv[1:] and sys.argv[1] == "file":
        input_stream = AudioFileInputStream(
            file_path="aiteacher/audio/start_stop.mp3",
        )
        input_stream.start()
        print("Audio file stream started.")
    else:
        input_stream = MicrophoneInputStream(sample_rate=16000)
        input_stream.start()
        print("Microphone stream started. Press Ctrl+C to stop.")

    audio_parser = AudioParser(model_path="vosk-model-small-en-us-0.15", sample_rate=16000)

    output_stream = sd.OutputStream(
        samplerate=16000, 
        # blocksize=2048,
        channels=1, 
        dtype='float32',
    )
    output_stream.start()

    prev_status = None
    try:
        while True:
            time.sleep(0.5)
            chunk = input_stream.get_unprocessed_chunk()
            if chunk is None:
                continue
            
            status, speech = audio_parser.add_chunk(chunk)
            if status != prev_status:
                print(f"Parser status: {status}")
                prev_status = status

            if status == 'listening':
                print(".", end="", flush=True)

            if speech is not None:
                print("\nSpeech interval detected. Transcribing...")
                transcription = speech_to_text(speech, language='en')
                print(f"Transcription: {transcription}")

                reply = answer(transcription)
                print(f"LLM Reply: {reply}")

                # Optionally, convert text back to speech
                tts_audio = text_to_speech(reply)
                output_stream.write(tts_audio)

                print(f"Generated TTS audio of length: {len(tts_audio)} bytes")



    
    
    except KeyboardInterrupt:
        print("Stopping microphone stream...")
    finally:
        input_stream.stop()
        output_stream.stop()
        output_stream.close()
        print("Microphone stream stopped.")




if __name__ == "__main__":
    main()
