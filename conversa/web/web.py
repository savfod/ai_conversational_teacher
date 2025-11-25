import os

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__, template_folder=os.path.dirname(__file__))
socketio = SocketIO(app, cors_allowed_origins="*")


# Receive audio from browser
@socketio.on("audio_in")
def handle_audio_in(data):
    # data is raw audio bytes (ArrayBuffer from JS)
    # Process it / forward it somewhere / store / analyze
    # For now: echo back to browser
    emit("audio_out", data, broadcast=False)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=5555, debug=True)

# Set the Flask template folder to this package's directory so
# `render_template('index.html')` will find `conversa/web/index.html`.
