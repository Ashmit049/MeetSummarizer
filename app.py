from flask import Flask, render_template, request, jsonify
import sounddevice as sd
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
from transformers import pipeline, MarianMTModel, MarianTokenizer
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

print("Loading models...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
whisper_model = WhisperModel("base", device="cpu")
print("Models loaded.")

recording_data = np.empty((0, 1), dtype=np.float32)
is_recording = False

SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'fr': 'French',
    'es': 'Spanish',
    'de': 'German'
}

def load_translation_model(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

def translate(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def start_recording_thread(duration=10, fs=44100):
    global recording_data, is_recording
    recording_data = np.empty((0, 1), dtype=np.float32)
    is_recording = True

    def record_thread():
        global recording_data, is_recording
        try:
            print(f"Recording via InputStream for {duration} seconds...")
            with sd.InputStream(samplerate=fs, channels=1, callback=audio_callback):
                while is_recording and len(recording_data) < fs * duration:
                    time.sleep(0.1)
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
    segments, info = whisper_model.transcribe(filename)
    transcribed_text = " ".join([segment.text for segment in segments])
    return transcribed_text, info.language

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def summarize_text(full_text, chunk_size=500):
    print("Summarizing text...")
    chunks = chunk_text(full_text, chunk_size)
    all_summaries = []
    for chunk in chunks:
        formatted_input = "summarize: " + chunk
        summary = summarizer(formatted_input, max_length=300, min_length=150, do_sample=False)
        all_summaries.append(summary[0]['summary_text'])
    return "\n\n".join(all_summaries)

def process_pipeline(file_path, user_lang=None):
    try:
        transcript_text, detected_lang = transcribe_audio(file_path)
        output_lang = user_lang or detected_lang

        if detected_lang != 'en':
            try:
                to_en_model, to_en_tokenizer = load_translation_model(detected_lang, 'en')
                transcript_text_en = translate(transcript_text, to_en_model, to_en_tokenizer)
            except:
                transcript_text_en = transcript_text
        else:
            transcript_text_en = transcript_text

        summary_en = summarize_text(transcript_text_en)

        if output_lang != 'en':
            try:
                back_model, back_tokenizer = load_translation_model('en', output_lang)
                summary_translated = translate(summary_en, back_model, back_tokenizer)
            except:
                summary_translated = summary_en
        else:
            summary_translated = summary_en

        return transcript_text, summary_translated, detected_lang, output_lang

    except Exception as e:
        print(f"Error in process_pipeline: {e}")
        return None, None, None, None

@app.route('/')
def index():
    return render_template('index.html', supported_languages=SUPPORTED_LANGUAGES)

@app.route('/start_recording', methods=['POST'])
def start_recording_route():
    data = request.json
    duration = int(data.get('duration', 10))
    start_recording_thread(duration=duration)
    return jsonify({"status": "recording_started", "duration": duration})

@app.route('/stop_recording', methods=['POST'])
def stop_recording_route():
    filename = os.path.join(app.config['UPLOAD_FOLDER'], 'recorded.wav')
    success = stop_recording(filename=filename)
    if success:
        transcript_text, summary_text = transcribe_audio(filename)
        summary_result = summarize_text(transcript_text)
        return jsonify({
            "transcript": transcript_text,
            "summary": summary_result,
            "detected_lang": "en",
            "output_lang": "en"
        })
    else:
        return jsonify({"error": "No audio recorded"}), 400

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio_file' not in request.files:
        return render_template('index.html', error="No file part", supported_languages=SUPPORTED_LANGUAGES)
    file = request.files['audio_file']
    user_lang = request.form.get('target_lang')
    if file.filename == '':
        return render_template('index.html', error="No selected file", supported_languages=SUPPORTED_LANGUAGES)
    if file:
        filename = werkzeug.utils.secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        transcript_text, summary_translated, detected_lang, output_lang = process_pipeline(file_path, user_lang)
        return render_template('index.html', transcript=transcript_text, summary=summary_translated, detected_lang=detected_lang, output_lang=output_lang, supported_languages=SUPPORTED_LANGUAGES)

if __name__ == '__main__':
    app.run(debug=True)
