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


---

---

## ⚠️ Challenges Faced

### 1. Distinguishing Silence from Low-Energy Speech
- Initial energy-based methods incorrectly detected soft speech as silence  
- Solved using `librosa.effects.split()` for better segmentation  

---

### 2. Detecting Meaningful Repetitions
- Not all similar segments are stutters  
- Tuned similarity threshold to reduce false positives  

---

### 3. Mapping Audio Segments to Words
- Aligning audio segments with transcribed words was challenging  
- Used approximate mapping based on segment index  

---

### 4. Phoneme Estimation
- Accurate phoneme extraction is complex  
- Used spectral centroid as a heuristic approximation  

---

### 5. Deployment Issues
- Missing dependencies (e.g., SpeechRecognition) caused errors  
- Fixed using proper `requirements.txt` configuration  

---

## 💡 Future Improvements
- Use deep learning models (e.g., wav2vec) for better phoneme detection  
- Improve word-level alignment using forced alignment techniques  
- Add waveform visualization  
- Support real-time microphone input  

---

## 📜 License
MIT License
