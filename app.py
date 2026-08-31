import streamlit as st
import numpy as np
import librosa
from tensorflow import keras
import joblib

st.set_page_config(
    page_title="Emotion from Voice",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url("https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap");

    html, body, [class*="css"] {
        font-family: "Nunito", sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #fff7ed 0%, #ffedd5 100%);
    }

    #MainMenu, footer, header {visibility: hidden;}

    .hero {
        text-align: center;
        padding: 2rem 0 0.5rem 0;
    }
    .hero-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .hero h1 {
        font-size: 2.2rem;
        font-weight: 900;
        color: #9a3412;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #c2410c;
        font-size: 1rem;
        font-weight: 600;
        opacity: 0.75;
    }

    .card {
        background: #ffffff;
        border-radius: 22px;
        padding: 1.8rem;
        margin-top: 1.2rem;
        box-shadow: 0 10px 30px rgba(154, 52, 18, 0.08);
        border: 2px solid #ffedd5;
    }

    .card-title {
        font-size: 1rem;
        font-weight: 800;
        color: #9a3412;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #fb923c, #f97316);
        color: white;
        font-weight: 800;
        font-size: 1.05rem;
        padding: 0.75rem;
        border-radius: 14px;
        border: none;
        box-shadow: 0 6px 18px rgba(249, 115, 22, 0.35);
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 22px rgba(249, 115, 22, 0.5);
        color: white;
    }

    .result-hero {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .result-emoji {
        font-size: 4.5rem;
        line-height: 1;
    }
    .result-label {
        font-size: 2rem;
        font-weight: 900;
        color: #9a3412;
        text-transform: capitalize;
        margin-top: 0.3rem;
    }
    .result-conf {
        color: #c2410c;
        font-weight: 700;
        font-size: 0.95rem;
        opacity: 0.8;
    }

    .prob-row {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin: 0.5rem 0;
    }
    .prob-emoji {
        font-size: 1.3rem;
        width: 28px;
    }
    .prob-name {
        width: 90px;
        font-weight: 700;
        color: #7c2d12;
        font-size: 0.85rem;
        text-transform: capitalize;
    }
    .prob-track {
        flex: 1;
        background: #ffedd5;
        border-radius: 10px;
        height: 14px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #fdba74, #f97316);
    }
    .prob-pct {
        width: 45px;
        text-align: right;
        font-size: 0.8rem;
        font-weight: 700;
        color: #9a3412;
    }

    .placeholder {
        text-align: center;
        padding: 2.5rem 1rem;
        color: #c2703c;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .footer-note {
        text-align: center;
        color: #c2703c;
        font-size: 0.78rem;
        margin-top: 2rem;
        padding-bottom: 1rem;
        font-weight: 600;
        opacity: 0.7;
    }

    [data-testid="stFileUploader"] {
        background: #fff7ed;
        border-radius: 14px;
        padding: 0.8rem;
        border: 2px dashed #fdba74;
    }
    </style>
""", unsafe_allow_html=True)

model = keras.models.load_model("emotion_recognition_model.h5")
label_encoder = joblib.load("label_encoder.pkl")

EMOTION_EMOJIS = {
    "angry": "😠", "calm": "😌", "disgust": "🤢", "fearful": "😨",
    "happy": "😄", "neutral": "😐", "sad": "😢", "surprised": "😲"
}

def extract_features(audio, sample_rate, max_len=150):
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=20)
    features = np.vstack([mfcc, chroma, mel])
    if features.shape[1] < max_len:
        pad_width = max_len - features.shape[1]
        features = np.pad(features, pad_width=((0,0),(0,pad_width)), mode="constant")
    else:
        features = features[:, :max_len]
    return features

st.markdown("""
    <div class="hero">
        <div class="hero-icon">🎧</div>
        <h1>What does your voice say?</h1>
        <p>Upload a voice clip and discover the emotion behind it</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<div class=\'card\'>", unsafe_allow_html=True)
st.markdown("<div class=\'card-title\'>🎵 Your Audio</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"], label_visibility="collapsed")
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
predict_clicked = st.button("🔍 Analyze Emotion")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class=\'card\'>", unsafe_allow_html=True)

if predict_clicked and uploaded_file is not None:
    audio, sample_rate = librosa.load(uploaded_file, sr=22050)
    feat = extract_features(audio, sample_rate)
    feat = np.transpose(feat, (1, 0))
    feat = feat.reshape(1, feat.shape[0], feat.shape[1])

    prediction = model.predict(feat, verbose=0)[0]
    predicted_class = np.argmax(prediction)
    predicted_emotion = label_encoder.classes_[predicted_class]
    confidence = float(np.max(prediction) * 100)
    emoji = EMOTION_EMOJIS.get(predicted_emotion, "")

    st.markdown(f"""
        <div class="result-hero">
            <div class="result-emoji">{emoji}</div>
            <div class="result-label">{predicted_emotion}</div>
            <div class="result-conf">{confidence:.1f}% confidence</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class=\'card-title\' style=\'margin-top:1rem;\'>📊 Full Breakdown</div>", unsafe_allow_html=True)

    sorted_pairs = sorted(zip(label_encoder.classes_, prediction), key=lambda x: x[1], reverse=True)

    for emotion, prob in sorted_pairs:
        pct = prob * 100
        emo_icon = EMOTION_EMOJIS.get(emotion, "")
        st.markdown(f"""
            <div class="prob-row">
                <div class="prob-emoji">{emo_icon}</div>
                <div class="prob-name">{emotion}</div>
                <div class="prob-track"><div class="prob-fill" style="width:{pct}%;"></div></div>
                <div class="prob-pct">{pct:.0f}%</div>
            </div>
        """, unsafe_allow_html=True)

elif predict_clicked:
    st.warning("Please upload an audio file first.")
else:
    st.markdown("""
        <div class="placeholder">
            🎙️ Upload a .wav file above, then tap<br><b>Analyze Emotion</b> to see the results here.
        </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
    <div class="footer-note">
        Powered by an LSTM neural network trained on the RAVDESS dataset
    </div>
""", unsafe_allow_html=True)
