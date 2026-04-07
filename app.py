import streamlit as st
import librosa
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import tempfile
import os

st.set_page_config(page_title="Speech Analysis", layout="wide")

st.title("🎧 Speech Analysis: Pause & Repetition Detection")

# -----------------------------
# AUDIO SOURCE SELECTION
# -----------------------------
st.sidebar.header("Audio Source")

option = st.sidebar.radio(
    "Choose input type:",
    ["Upload Audio", "Use LibriSpeech Sample", "Use UCLASS Sample"]
)

audio_path = None

# Upload option
if option == "Upload Audio":
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            audio_path = tmp.name

# Dataset samples
elif option == "Use LibriSpeech Sample":
    audio_path = "data/librispeech_sample.wav"

elif option == "Use UCLASS Sample":
    audio_path = "data/uclass_sample.wav"


# -----------------------------
# PROCESS ONLY IF AUDIO EXISTS
# -----------------------------
if audio_path is not None and os.path.exists(audio_path):

    st.audio(audio_path)

    # -----------------------------
    # LOAD AUDIO
    # -----------------------------
    y, sr = librosa.load(audio_path, sr=None)

    # -----------------------------
    # PREPROCESSING
    # -----------------------------
    st.header("🧹 Preprocessing")

    y = y / np.max(np.abs(y))
    y = librosa.effects.preemphasis(y)

    st.write("✔️ Audio normalized and pre-emphasized")

    # -----------------------------
    # PAUSE DETECTION
    # -----------------------------
    st.header("⏸️ Pause Detection")

    frame_length = 2048
    hop_length = 512

    energy = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length
    )[0]

    times = librosa.frames_to_time(
        range(len(energy)), sr=sr, hop_length=hop_length
    )

    threshold = st.slider("Silence Threshold", 0.0, 0.1, 0.02)

    silent_frames = energy < threshold

    pauses = []
    start = None

    for i, is_silent in enumerate(silent_frames):
        if is_silent and start is None:
            start = times[i]
        elif not is_silent and start is not None:
            end = times[i]
            pauses.append((start, end))
            start = None

    if start is not None:
        pauses.append((start, times[-1]))

    total_pause = sum(end - start for start, end in pauses)

    st.subheader("Detected Pause Segments")
    if pauses:
        for start, end in pauses:
            st.write(f"[{start:.2f}s - {end:.2f}s]")
    else:
        st.write("No pauses detected.")

    st.write(f"**Total Pause Duration:** {total_pause:.2f}s")

    # Plot energy
    fig, ax = plt.subplots()
    ax.plot(times, energy)
    ax.axhline(threshold, linestyle='--')
    ax.set_title("Energy vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy")
    st.pyplot(fig)

    # -----------------------------
    # REPETITION DETECTION
    # -----------------------------
    st.header("🔁 Repetition Detection")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    segment_size = st.slider("Segment Size (frames)", 5, 50, 20)
    similarity_threshold = st.slider("Similarity Threshold", 0.7, 1.0, 0.9)

    segments = []
    segment_times = []

    for i in range(0, mfcc.shape[1] - segment_size, segment_size):
        segment = mfcc[:, i:i + segment_size]
        segments.append(segment)

        t_start = librosa.frames_to_time(i, sr=sr)
        t_end = librosa.frames_to_time(i + segment_size, sr=sr)
        segment_times.append((t_start, t_end))

    repetitions = []

    for i in range(len(segments) - 1):
        s1 = segments[i].flatten()
        s2 = segments[i + 1].flatten()

        sim = 1 - cosine(s1, s2)

        if sim > similarity_threshold:
            repetitions.append((segment_times[i], segment_times[i + 1], sim))

    st.subheader("Detected Repetitions")

    if len(repetitions) == 0:
        st.write("No repetitions detected.")
    else:
        for (t1, t2, sim) in repetitions:
            st.write(
                f"[{t1[0]:.2f}s - {t1[1]:.2f}s] ≈ "
                f"[{t2[0]:.2f}s - {t2[1]:.2f}s] "
                f"(Similarity: {sim:.2f})"
            )

    st.write(f"**Repetition Count:** {len(repetitions)}")

    # -----------------------------
    # FINAL SUMMARY
    # -----------------------------
    st.header("📊 Final Output Summary")

    st.write("### Pause Segments:")
    for start, end in pauses:
        st.write(f"[{start:.2f}s – {end:.2f}s]")

    st.write(f"Total Pause Duration: {total_pause:.2f}s")

    st.write("### Repetitions:")
    st.write(f"Repetition Count: {len(repetitions)}")

else:
    st.warning("Please upload or select an audio file.")
