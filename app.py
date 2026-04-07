import streamlit as st
import librosa
import numpy as np
from scipy.spatial.distance import cosine
import tempfile
import os
import speech_recognition as sr
import requests

# -----------------------------
# PAGE CONFIG + STYLE
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
}
.section-title {
    font-size: 20px;
    font-weight: 600;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown('<div class="main-title">🎧 Speech Analysis System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Pause Detection & Stutter Recognition</div>', unsafe_allow_html=True)

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
# SIDEBAR INPUT
# -----------------------------
st.sidebar.header("Audio Source")

option = st.sidebar.radio(
    "Choose input:",
    ["Upload Audio", "Use Sample Audio"]
)

audio_path = None

# -----------------------------
# OPTION 1: UPLOAD
# -----------------------------
if option == "Upload Audio":
    uploaded_file = st.sidebar.file_uploader("Upload WAV file", type=["wav"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            audio_path = tmp.name

# -----------------------------
# OPTION 2: SAMPLE FROM GITHUB
# -----------------------------
elif option == "Use Sample Audio":

    if st.sidebar.button("Load Sample Audio"):

        url = "https://raw.githubusercontent.com/akeren12/Audio-Pause-And-Repetition-Detection/main/data/M_0030_16y4m_1.wav"

        response = requests.get(url)

        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(response.content)
                audio_path = tmp.name

            st.success("Sample audio loaded successfully!")
        else:
            st.error("Failed to load sample audio.")

# -----------------------------
# MAIN PROCESS
# -----------------------------
if audio_path is not None:

    st.audio(audio_path)

    if st.button("🚀 Run Analysis"):

        with st.spinner("Analyzing audio..."):

            y, sr_rate = librosa.load(audio_path, sr=None)
            y = y / np.max(np.abs(y))

            # -----------------------------
            # PAUSE DETECTION (>= 2 sec)
            # -----------------------------
            intervals = librosa.effects.split(y, top_db=25)

            pauses = []
            for i in range(len(intervals) - 1):
                end_current = intervals[i][1] / sr_rate
                start_next = intervals[i + 1][0] / sr_rate

                if (start_next - end_current) >= 2.0:
                    pauses.append((end_current, start_next))

            total_pause = sum(e - s for s, e in pauses)

            # -----------------------------
            # REPETITION DETECTION
            # -----------------------------
            mfcc = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=13)

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

            stuttered_words = build_stutter_patterns(words, repetition_indices, y, sr_rate)

        # -----------------------------
        # OUTPUT UI
        # -----------------------------
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown('<div class="section-title">📄 File</div>', unsafe_allow_html=True)
        st.write(os.path.basename(audio_path))

        st.markdown('<div class="section-title">⏸️ Pause Segments</div>', unsafe_allow_html=True)

        if pauses:
            for s, e in pauses:
                st.write(f"[{s:.1f}s – {e:.1f}s]")
        else:
            st.write("None")

        st.write(f"**Total Pause Duration:** {total_pause:.1f}s")

        st.markdown('<div class="section-title">🔁 Stuttered Words</div>', unsafe_allow_html=True)

        if stuttered_words:
            st.write(", ".join(stuttered_words))
            st.write(f"**Count:** {len(stuttered_words)}")
        else:
            st.write("None")
            st.write("Count: 0")

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Upload or load sample audio to begin.")
