# 🎧 Speech Analysis: Pause & Repetition Detection

## 🌐 Live Demo
🚀 Try the app here:  
https://audio-pause-and-repetition-detection.streamlit.app/

---

## 🧪 Sample Audio

A sample audio file is included in the repository under the data folder.  
Kindly download the file and upload it in the website.


---

## 📌 Overview
This project is a real-time speech analysis system that detects:
- ⏸️ Pause Segments (silent regions in speech)
- 🔁 Stuttered Speech Patterns (e.g., *ba-ba-ball*, *gi-gi-give*)

The system combines signal processing and phoneme-based reconstruction to analyze speech and generate structured outputs.

---

## 🎯 Objective
To design an explainable pipeline that:
- Detects pauses in speech
- Identifies stuttering patterns
- Outputs meaningful and human-readable results

---

## ⚙️ Tech Stack
- **Python**
- **Streamlit** – Web UI  
- **Librosa** – Audio processing  
- **NumPy** – Numerical operations  
- **SciPy** – Similarity computation  
- **SpeechRecognition** – Speech-to-text  

---

## 🧠 Approach

### 1️⃣ Audio Preprocessing
- Normalize audio signal
- Apply pre-emphasis to enhance speech features

---

### 2️⃣ Pause Detection
- Uses `librosa.effects.split()` to detect speech regions
- Pauses are calculated as gaps between speech segments
- Only pauses **≥ 2 seconds** are considered

This avoids misclassifying low-energy speech as silence.

---

### 3️⃣ Repetition Detection

#### Feature Extraction
- Extract MFCC (Mel-Frequency Cepstral Coefficients)

#### Segment Comparison
- Split audio into segments
- Compare consecutive segments using cosine similarity

#### Detection
- High similarity → repeated speech pattern

---

### 4️⃣ Phoneme-Based Stutter Reconstruction
- Extract repeated audio segments
- Estimate phoneme using spectral features
- Combine with recognized words

Example:
