import argparse
import time
from threading import Thread

import numpy as np

from conversa.features.llm_api import call_llm
from conversa.generated.speech_api import speech_to_text, text_to_speech
from conversa.web import server
from conversa.web.io import WebInputStream, WebOutputStream

CHUNK_SIZE = 16000 * 5  # e.g. 5 second @ 16kHz


def process_audio(full_audio: np.ndarray, debug: bool = False) -> np.ndarray | None:
    """Process full audio chunk and return processed audio.
    Args:
        full_audio: NumPy array of shape (n,) dtype float32.
        debug: If True, prints debug information.
    Returns:
        Processed audio as NumPy array of shape (n,) dtype float32.
    """
    if debug:
        time.sleep(1)  # Simulate processing delay
        return full_audio

    text = speech_to_text(full_audio, sample_rate=16000, language="en")
    if text != "" and not text.strip().startswith(
        "Please transcribe the following audio"
    ):
        print(f"processing text '{text}'")
        answer = call_llm(text, sys_prompt="You are a helpful assistant.")
        # TODO: Replace with your logic / ML model / filtering
        return text_to_speech(answer)  # For now: identity
    else:
        print(f"skipped text '{text}'")
        return None


def audio_worker(debug: bool = False):
    """
    Continuously collects audio chunks from WebInputStream.
    Accumulates enough samples → process_audio() → send back using WebOutputStream.

    Args:
        debug: If True, prints debug information.
    """
    # Initialize streams
    input_stream = WebInputStream(sample_rate=16000, channels=1)
    output_stream = WebOutputStream(sample_rate=16000, channels=1)

    input_stream.start()
    # Output stream doesn't need start() strictly but it's good practice if it did
    # output_stream.start()

    print("Streams started. Waiting for audio...")

    buffer = np.array([], dtype=np.float32)

    try:
        while True:
            # Get new data
            chunk = input_stream.get_unprocessed_chunk()
            if chunk is not None:
                buffer = np.concatenate((buffer, chunk))

            # Process if enough data
            if len(buffer) >= CHUNK_SIZE:
                to_process = buffer[:CHUNK_SIZE]
                buffer = buffer[CHUNK_SIZE:]  # Keep remainder

                # Run your processing
                processed = process_audio(to_process, debug=debug)

                if processed is not None:
                    # Send back
                    output_stream.play_chunk(processed)

            # Small sleep to avoid busy loop if no data
            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        input_stream.stop()
        output_stream.stop()


def arg_parser():
    parser = argparse.ArgumentParser(description="Conversa Web Server")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()

    # Start the worker logic in a separate thread
    Thread(target=audio_worker, daemon=True, args=(args.debug,)).start()

    # Run the server
    # Note: We run this in the main thread as it blocks
    server.run_server(host="127.0.0.1", port=5557, debug=True)
