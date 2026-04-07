# 🎧 Speech Analysis: Pause & Repetition Detection

## 📌 Overview

This project implements a real-time speech analysis system that detects:

* ⏸️ **Pause Segments** (silent regions in speech)
* 🔁 **Stuttered Speech Patterns** (e.g., *ba-ba-ball*, *I-I-I*)

The system uses audio signal processing techniques and phoneme-based reconstruction to analyze speech and produce structured outputs.

---

## 🎯 Objective

To design an explainable pipeline that:

* Identifies pauses in speech
* Detects repetition patterns caused by stuttering
* Outputs results in a clean, structured format

---

## ⚙️ Tech Stack

* **Language:** Python
* **Libraries:**

  * `librosa` – audio processing
  * `numpy` – numerical computation
  * `scipy` – similarity calculation
  * `speech_recognition` – speech-to-text
  * `streamlit` – UI interface

---

## 🧠 Approach

### 1️⃣ Audio Preprocessing

* Audio is normalized to ensure consistent amplitude
* Pre-emphasis applied to enhance speech clarity

---

### 2️⃣ Pause Detection

* Used `librosa.effects.split()` to detect **non-silent regions**
* Pauses are calculated as gaps between speech segments
* Only pauses **≥ 2 seconds** are considered

✔️ This avoids misclassifying low-energy speech as silence

---

### 3️⃣ Repetition Detection (Stuttering)

#### 🔹 Step 1: Feature Extraction

* Extract **MFCC (Mel-Frequency Cepstral Coefficients)**

#### 🔹 Step 2: Segment Comparison

* Split audio into small segments
* Compare consecutive segments using **cosine similarity**

#### 🔹 Step 3: Detect Repetitions

* High similarity between segments → indicates repetition

---

### 4️⃣ Phoneme-Based Reconstruction

* Extract short repeated audio segments
* Estimate phoneme using **spectral features**
* Combine with transcribed words to reconstruct patterns

Example:

```
ba-ba-ball
gi-gi-give
ca-ca-cat
```

---

## 🧪 Output Format

```
File: sample.wav

Pause Segments:
[20.1s – 23.9s]

Total Pause Duration: 3.7s

Stuttered Words:
ba-ba-ball, gi-gi-give

Count: 2
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/speech-analysis.git
cd speech-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

### 4. Upload a `.wav` file and click **Run Analysis**

---

## 📁 Project Structure

```
speech-analysis/
│
├── app.py
├── requirements.txt
├── data/
│   ├── sample_audio.wav
└── README.md
```

---

## 📊 Datasets Used

* LibriSpeech Dataset (for pause detection)
* UCLASS Stuttered Speech Dataset (for repetition detection)

---

## ⚠️ Limitations

* Speech recognition may normalize stuttered speech
* Phoneme detection is approximate (heuristic-based)
* Accuracy depends on audio quality

---

## 💡 Future Improvements

* Use deep learning models (e.g., wav2vec) for phoneme detection
* Improve alignment between audio segments and words
* Add visualization of waveform and pause regions
* Real-time microphone input support

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👤 Author

Developed as part of an internship assignment on speech analysis and signal processing.
