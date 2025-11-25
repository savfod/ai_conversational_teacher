import os
import queue
import sys
from threading import Thread

import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO

from conversa.features.llm_api import call_llm
from conversa.generated.speech_api import speech_to_text, text_to_speech

app = Flask(__name__, template_folder=os.path.dirname(__file__))
socketio = SocketIO(app, cors_allowed_origins="*")

# Queue of (sid, audio_bytes)
audio_queue = queue.Queue()

# Accumulation buffer PER CLIENT
buffers = {}  # sid -> list of pcm chunks

SAMPLE_WIDTH = 2  # int16
CHUNK_SIZE = 16000 * 5  # e.g. 5 second @ 16kHz


# -----------------------------
#  Your audio processing logic
# -----------------------------
def process_audio(full_audio: np.ndarray) -> np.ndarray:
    """Process full audio chunk and return processed audio.
    Args:
        full_audio: NumPy array of shape (n,) dtype float32.
    Returns:
        Processed audio as NumPy array of shape (n,) dtype float32.
    """
    text = speech_to_text(full_audio, sample_rate=16000, language="en")
    answer = call_llm(text, sys_prompt="You are a helpful assistant.")
    # TODO: Replace with your logic / ML model / filtering
    return text_to_speech(answer)  # For now: identity


# -----------------------------
#     WebSocket Handlers
# -----------------------------
@socketio.on("audio_in")
def handle_audio_in(data):
    """
    Receives raw audio bytes from browser.
    Stores (sid, bytes) in processing queue.
    """
    sid = request.sid
    audio_queue.put((sid, data))


# -----------------------------
#        Worker Thread
# -----------------------------
def audio_worker():
    """
    Continuously collects audio chunks from queue.
    Accumulates enough samples → process_audio() → send back.
    """
    while True:
        sid, chunk = audio_queue.get()

        # Convert raw bytes to PCM16 NumPy
        pcm = np.frombuffer(chunk, dtype=np.int16)

        # Per-client buffer
        if sid not in buffers:
            buffers[sid] = []

        buffers[sid].append(pcm)

        # Total accumulated samples
        total = np.concatenate(buffers[sid])

        # Process only when enough audio accumulated
        if len(total) >= CHUNK_SIZE:
            to_process = total[:CHUNK_SIZE]

            # Run your processing
            processed = process_audio(to_process)

            # Convert back to bytes
            out_bytes = processed.astype(np.int16).tobytes()

            # Send back to EXACT client (full‑duplex)
            socketio.emit("audio_out", out_bytes, room=sid)

            # Retain leftover samples
            buffers[sid] = [total[CHUNK_SIZE:]]


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    Thread(target=audio_worker, daemon=True).start()  # type: ignore
    socketio.run(app, host="127.0.0.1", port=5555, debug=True)

# Set the Flask template folder to this package's directory so
# `render_template('index.html')` will find `conversa/web/index.html`.
