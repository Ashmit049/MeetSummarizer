🧠 Real-Time Meeting Minutes Generator
A web-based tool to record or upload audio, transcribe it using Whisper, and generate smart summaries using BART. Supports multilingual summaries for uploads and reliable English summaries for recordings — all within a sleek, tab-based UI.

🚀 Features
🎙️ Real-time audio recording from browser (English-only)

📤 Upload audio files in multiple languages

🧠 Transcription using Faster Whisper

✂️ Summarization using facebook/bart-large-cnn

🌍 Multilingual support: auto-translate transcript and summary

📋 Copy or download summaries

⚡ Fast and lightweight — runs on CPU or GPU

🧾 Responsive UI with tab-based interaction

🧪 How It Works
🔴 Record Tab (English Only)
Choose recording duration (30 sec to 10 min)

Record directly in the browser using your microphone

Transcription and summarization are done in English

Results are displayed in real time

📤 Upload Tab (Multilingual)
Upload an audio file in any supported language

Choose the target language for the summary

Pipeline:

Auto-detect language

Translate transcript → English

Summarize in English

Translate summary → selected language

View transcript and summary in your chosen language
