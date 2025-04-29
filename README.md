# 🧠 Real-Time Meeting Minutes Generator

A web-based application to **record or upload audio**, transcribe it using **Whisper**, and generate smart summaries using **BART**.  
Supports **multilingual summaries** for uploads and **reliable English summaries** for recordings — all within a clean, interactive UI.

---

## 🚀 Features

- 🎙️ Real-time audio recording (English-only)
- 📤 Upload audio files (supports multiple languages)
- 🧠 Transcription powered by [Faster Whisper](https://github.com/guillaumekln/faster-whisper)
- ✂️ Summarization using `facebook/bart-large-cnn`
- 🌍 Multilingual support: auto-translate transcripts and summaries
- 📋 Copy or download summaries
- ⚡ Fast and lightweight (CPU/GPU support)
- 🧾 Responsive UI with tab-based navigation

---

## 🧪 How It Works

### 🔴 Record Tab (English Only)

1. Choose a recording duration
2. Record directly in the browser
3. Get real-time transcription and summary in English
4. View and copy/download results

### 📤 Upload Tab (Multilingual)

1. Upload an audio file in any supported language
2. Choose a target language for the summary
3. The system:
   - Detects the source language
   - Translates transcript to English
   - Summarizes it in English
   - Translates the summary to your chosen language
4. View translated transcript and summary

---

## 🌐 Supported Upload Languages

- 🇬🇧 English (`en`)
- 🇮🇳 Hindi (`hi`)
- 🇫🇷 French (`fr`)
- 🇪🇸 Spanish (`es`)
- 🇩🇪 German (`de`)



