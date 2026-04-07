# 🎧 Speech Analysis: Pause & Repetition Detection

## 🌐 Live Demo
https://audio-pause-and-repetition-detection.streamlit.app/

---

## 📌 Objective
To build a system that analyzes speech audio and detects:
- ⏸️ Pause segments (silent regions)
- 🔁 Repetition patterns (stuttered speech)

---

## ⚙️ Features Used
- **Audio Processing** using `librosa`
- **MFCC (Mel-Frequency Cepstral Coefficients)** for feature extraction
- **Cosine Similarity** for detecting repeated segments
- **SpeechRecognition** for converting speech to text
- **Streamlit UI** for interactive usage
- **GitHub-hosted sample audio integration**

---

## 🧠 Approach

### 1. Audio Preprocessing
- Audio signal is normalized to maintain consistent amplitude
- This helps improve feature extraction and detection stability

---

### 2. Pause Detection
- Used `librosa.effects.split()` to identify **non-silent (speech) segments**
- Pauses are computed as gaps between these segments

#### Logic:
- If gap between two speech segments ≥ 2 seconds → classified as pause

This method avoids misclassifying low-energy speech as silence.

---

### 3. Repetition Detection (Stuttering)

#### Step 1: Feature Extraction
- Extract MFCC features from audio

#### Step 2: Segmentation
- Audio is divided into small overlapping segments

#### Step 3: Similarity Comparison
- Consecutive segments are compared using cosine similarity

#### Logic:
- If similarity > threshold (≈ 0.92) → repetition detected

---

### 4. Stuttered Word Reconstruction
- Detected repeated segments are mapped to transcribed words
- A simple phoneme estimation is performed using spectral features
- Output is formatted as:
