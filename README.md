# 🎧 Speech Analysis: Pause & Repetition Detection

## 🌐 Live Demo
🚀 Try the app here:  
https://audio-pause-and-repetition-detection.streamlit.app/

---

## 📌 Overview
This project is a real-time speech analysis system that detects:
- ⏸️ **Pause Segments** (silent regions in speech)
- 🔁 **Stuttered Speech Patterns** (e.g., *ba-ba-ball*, *gi-gi-give*)

The system uses audio signal processing and phoneme-based reconstruction to analyze speech and generate structured outputs.

---

## ✨ Key Features
- 🎧 Upload your own audio (.wav)
- 🎯 Built-in **sample audio (no download required)**
- ⏸️ Accurate pause detection (≥ 2 seconds)
- 🔁 Detection of stuttered/repeated speech patterns
- 🧠 Phoneme-based reconstruction of stuttered words
- ⚡ Clean and interactive Streamlit UI

---

## 🧠 Approach

### 1️⃣ Audio Preprocessing
- Normalize audio signal for consistency  
- Improve clarity using basic preprocessing  

---

### 2️⃣ Pause Detection
- Uses `librosa.effects.split()` to detect speech regions  
- Pauses calculated as gaps between speech  
- Only pauses ≥ 2 seconds are considered  

---

### 3️⃣ Repetition Detection
- Extract MFCC (Mel-Frequency Cepstral Coefficients)  
- Segment audio into small chunks  
- Compare consecutive segments using cosine similarity  
- High similarity → repeated speech  

---

### 4️⃣ Stutter Reconstruction
- Estimate phoneme from repeated audio segments  
- Combine with recognized words  
- Output patterns like:
