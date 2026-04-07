import streamlit as st
import librosa
import numpy as np
from scipy.spatial.distance import cosine
import tempfile
import os
import speech_recognition as sr

# -----------------------------
# PAGE CONFIG + CUSTOM STYLE
# -----------------------------
st.set_page_config(page_title="Speech Analysis", layout="wide")

st.markdown("""
<style>
body {
    font-family: 'Segoe UI', sans-serif;
}

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 10px;
}

.sub-title {
    text-align: center;
    color: #888;
    margin-bottom: 30px;
}

.card {
    background-color: #111827;
    padding: 25px;
    border-radius: 12px;
    margin-top: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}

.section-title {
    font-size: 20px;
    font-weight: 600;
    margin-top: 15px;
}

.output-text {
    font-size: 16px;
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown('<div class="main-title">🎧 Speech Analysis System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Pause Detection & Stutter Recognition using Audio Processing</div>', unsafe_allow_html=True)

# -----------------------------
# FUNCTIONS
# -----------------------------
def get_transcription(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        return ""

def estimate_phoneme(audio_segment, sr):
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio_segment, sr=sr))

    if centroid < 1500:
        return "b"
    elif centroid < 2500:
        return "d"
    elif centroid < 3500:
        return "g"
    elif centroid < 4500:
        return "k"
    else:
        return "s"

def build_stutter_patterns(words, repetition_indices, y, sr):
    stuttered = []

    for idx, rep_index in enumerate(repetition_indices):
        if idx >= len(words):
            break

        word = words[idx]

        start = int(rep_index * 512)
        end = start + int(0.3 * sr)

        segment = y[start:end]

        if len(segment) == 0:
            phoneme = word[0]
        else:
            phoneme = estimate_phoneme(segment, sr)

        pattern = f"{phoneme}{word[1:]}-{phoneme}{word[1:]}-{word}"
        stuttered.append(pattern)

    return stuttered

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Upload Audio")
uploaded_file = st.sidebar.file_uploader("Choose WAV file", type=["wav"])

audio_path = None

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

# -----------------------------
# MAIN UI
# -----------------------------
if audio_path is not None:

    st.audio(audio_path)

    if st.button("🚀 Run Analysis"):

        with st.spinner("Analyzing audio..."):

            y, sample_rate = librosa.load(audio_path, sr=None)
            y = y / np.max(np.abs(y))

            # -----------------------------
            # PAUSE DETECTION
            # -----------------------------
            intervals = librosa.effects.split(y, top_db=25)

            pauses = []
            for i in range(len(intervals) - 1):
                end_current = intervals[i][1] / sample_rate
                start_next = intervals[i + 1][0] / sample_rate

                if (start_next - end_current) >= 2.0:
                    pauses.append((end_current, start_next))

            total_pause = sum(end - start for start, end in pauses)

            # -----------------------------
            # REPETITION DETECTION
            # -----------------------------
            mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=13)

            segment_size = 30
            similarity_threshold = 0.92

            segments = [
                mfcc[:, i:i + segment_size]
                for i in range(0, mfcc.shape[1] - segment_size, segment_size)
            ]

            repetition_indices = []

            for i in range(len(segments) - 1):
                sim = 1 - cosine(segments[i].flatten(), segments[i + 1].flatten())
                if sim > similarity_threshold:
                    repetition_indices.append(i)

            repetition_indices = repetition_indices[:5]

            # -----------------------------
            # TEXT + STUTTER BUILD
            # -----------------------------
            text = get_transcription(audio_path)
            words = text.lower().split()

            stuttered_words = build_stutter_patterns(words, repetition_indices, y, sample_rate)

        # -----------------------------
        # OUTPUT UI
        # -----------------------------
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown('<div class="section-title">1. File Details</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="output-text">File: {os.path.basename(audio_path)}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">2. Pause Detection</div>', unsafe_allow_html=True)

        pause_str = ", ".join([f"[{s:.1f}s – {e:.1f}s]" for s, e in pauses]) if pauses else "None"

        st.markdown(f'<div class="output-text">{pause_str}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="output-text"><b>Total Pause Duration:</b> {total_pause:.1f}s</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">3. Stuttered Words</div>', unsafe_allow_html=True)

        if stuttered_words:
            stutter_str = ", ".join(stuttered_words)
            count = len(stuttered_words)
        else:
            stutter_str = "None"
            count = 0

        st.markdown(f'<div class="output-text">{stutter_str}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="output-text"><b>Count:</b> {count}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Upload an audio file to start analysis.")
