from flask import Flask, render_template, request, jsonify
import sounddevice as sd
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
from transformers import pipeline
import time
import os
import threading
import numpy as np
import werkzeug.utils
import traceback

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models once at startup
print("Loading models...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # Faster model
whisper_model = WhisperModel("base", device="cpu")  # Use CPU instead of GPU
print("Models loaded.")

recording_data = np.empty((0, 1), dtype=np.float32)
is_recording = False


def log_audio_devices():
    print("Available audio devices:")
    print(sd.query_devices())


def record_audio(filename="temp.wav", duration=10, fs=44100):
    print(f"Recording for {duration} seconds using blocking method.")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wav.write(filename, fs, recording)
    print("Recording complete.")
    return filename


def start_recording_thread(duration=10, fs=44100):
    global recording_data, is_recording
    recording_data = np.empty((0, 1), dtype=np.float32)
    is_recording = True

    def record_thread():
        global recording_data, is_recording
        try:
            print(f"Starting InputStream for {duration} seconds...")
            with sd.InputStream(samplerate=fs, channels=1, callback=audio_callback):
                while is_recording and len(recording_data) < fs * duration:
                    time.sleep(0.1)
            print("Recording thread completed.")
        except Exception as e:
            print("Error in recording thread:", e)
            traceback.print_exc()

    threading.Thread(target=record_thread).start()


def audio_callback(indata, frames, time_info, status):
    global recording_data
    if status:
        print(f"Audio status: {status}")
    try:
        if recording_data.size == 0:
            recording_data = indata.copy()
        else:
            recording_data = np.vstack((recording_data, indata))
        print(f"Callback received: {indata.shape} frames, total buffer: {recording_data.shape}")
    except Exception as e:
        print("Error in audio_callback:", e)
        traceback.print_exc()


def stop_recording(filename="temp.wav", fs=44100):
    global recording_data, is_recording
    is_recording = False
    time.sleep(0.5)
    if recording_data.shape[0] > 0:
        print(f"Saving recording with {recording_data.shape[0]} samples to {filename}")
        wav.write(filename, fs, recording_data)
        return True
    else:
        print("No audio data captured.")
        return False


def transcribe_audio(filename):
    print("Transcribing audio file: " + filename)
    start = time.time()
    segments, _ = whisper_model.transcribe(filename)
    end = time.time()
    print(f"Transcription completed in {end - start:.2f} seconds.")
    transcribed_text = " ".join([segment.text for segment in segments])
    return transcribed_text


def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def summarize_text(full_text, chunk_size=500):
    print("Chunking and summarizing transcript.")
    chunks = chunk_text(full_text, chunk_size)
    print(f"Total chunks: {len(chunks)}")

    all_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}")
        formatted_input = "summarize: " + chunk
        summary = summarizer(
            formatted_input,
            max_length=200,
            min_length=60,
            do_sample=False
        )
        all_summaries.append(summary[0]['summary_text'])

    return "\n\n".join(all_summaries)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start_recording', methods=['POST'])
def start_recording_route():
    log_audio_devices()
    data = request.json
    duration = int(data.get('duration', 10))
    start_recording_thread(duration=duration)
    return jsonify({"success": True, "message": f"Recording started for {duration} seconds"})


@app.route('/stop_recording', methods=['POST'])
def stop_recording_route():
    filename = "temp.wav"
    result = stop_recording(filename)
    if result:
        transcript_text = transcribe_audio(filename)
        summary_text = summarize_text(transcript_text)
        return render_template('index.html', transcript=transcript_text, summary=summary_text)
    else:
        return render_template('index.html', error="Recording failed or no input received.")


@app.route('/record', methods=['POST'])
def record():
    duration = int(request.form['duration'])
    filename = "temp.wav"
    record_audio(filename, duration)
    transcript_text = transcribe_audio(filename)
    summary_text = summarize_text(transcript_text)
    return render_template('index.html', transcript=transcript_text, summary=summary_text)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio_file' not in request.files:
        return render_template('index.html', error="No file part")
    file = request.files['audio_file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")
    if file:
        filename = werkzeug.utils.secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        transcript_text = transcribe_audio(file_path)
        summary_text = summarize_text(transcript_text)
        return render_template('index.html', transcript=transcript_text, summary=summary_text)


if __name__ == '__main__':
    app.run(debug=True)
