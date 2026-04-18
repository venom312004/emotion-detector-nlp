import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🧠",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f0f1a; color: #e8e8f0; }

    /* Title */
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #a78bfa, #60a5fa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }

    /* Text area */
    .stTextArea textarea {
        background-color: #1a1a2e !important;
        color: #e8e8f0 !important;
        border: 1.5px solid #2d2d4e !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #a78bfa !important;
        box-shadow: 0 0 0 2px rgba(167,139,250,0.2) !important;
    }

    /* Predict button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #7c3aed, #2563eb);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(124,58,237,0.4);
        color: white;
    }
    .stButton > button:active { transform: translateY(0); }

    /* Emotion result card */
    .emotion-card {
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1.5rem 0 1rem 0;
        text-align: center;
        animation: fadeIn 0.5s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .emotion-emoji  { font-size: 4rem; display: block; margin-bottom: 0.5rem; }
    .emotion-name   { font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem; }
    .emotion-conf   { font-size: 1rem; opacity: 0.75; }

    /* Metric style */
    .metric-box {
        background: #1a1a2e;
        border: 1px solid #2d2d4e;
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
    }
    .metric-val { font-size: 1.4rem; font-weight: 700; color: #a78bfa; }
    .metric-lbl { font-size: 0.78rem; color: #6b7280; margin-top: 2px; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #13131f;
        border-right: 1px solid #2d2d4e;
    }
    .sidebar-header {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6b7280;
        margin: 1rem 0 0.5rem;
    }

    /* Divider */
    hr { border-color: #2d2d4e; }

    /* Spinner color */
    .stSpinner > div { border-top-color: #a78bfa !important; }

    /* Hide default streamlit footer/header */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Emotion config ────────────────────────────────────────────────────────────
EMOTIONS = {
    "sadness":  {"emoji": "😢", "color": "#3b82f6", "bg": "#1e3a5f"},
    "anger":    {"emoji": "😠", "color": "#ef4444", "bg": "#4a1a1a"},
    "love":     {"emoji": "❤️",  "color": "#ec4899", "bg": "#4a1a35"},
    "surprise": {"emoji": "😲", "color": "#f59e0b", "bg": "#4a3a1a"},
    "fear":     {"emoji": "😨", "color": "#8b5cf6", "bg": "#2d1a4a"},
    "joy":      {"emoji": "😄", "color": "#22c55e", "bg": "#1a3a2a"},
}

LABEL_MAP = {0: "sadness", 1: "anger", 2: "love", 3: "surprise", 4: "fear", 5: "joy"}

# ── NLP cleaning helpers ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def download_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt",     quiet=True)
    return set(nltk.corpus.stopwords.words("english"))

def clean_text(txt, stop_words):
    txt = txt.lower()
    txt = txt.translate(str.maketrans("", "", string.punctuation))
    txt = re.sub(r"\d+", "", txt)
    txt = re.sub(r"https?://\S+|www\.\S+", "", txt)
    txt = "".join(c for c in txt if c.isascii())
    txt = " ".join(w for w in txt.split() if w not in stop_words)
    return txt

# ── Model training ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train(data_path):
    stop_words = download_nltk()

    df = pd.read_csv(data_path, sep=";", header=None, names=["text", "emotion"])
    df.dropna(inplace=True)
    df["emotion"] = df["emotion"].map(
        {"sadness": 0, "anger": 1, "love": 2, "surprise": 3, "fear": 4, "joy": 5}
    )
    df.dropna(subset=["emotion"], inplace=True)
    df["emotion"] = df["emotion"].astype(int)
    df["text"] = df["text"].apply(lambda x: clean_text(str(x), stop_words))

    X, y = df["text"], df["emotion"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tfidf = TfidfVectorizer(max_features=20000)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec  = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=5, solver="lbfgs", multi_class="auto")
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    counts = df["emotion"].value_counts().to_dict()
    dist = {LABEL_MAP[k]: int(v) for k, v in counts.items() if k in LABEL_MAP}

    return tfidf, model, acc, len(df), dist

def predict_emotion(text, tfidf, model, stop_words):
    cleaned = clean_text(text, stop_words)
    vec = tfidf.transform([cleaned])
    pred_label = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    emotion_name = LABEL_MAP[pred_label]
    scores = {LABEL_MAP[i]: float(proba[i]) for i in range(len(proba))}
    return emotion_name, scores

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Emotion Detector")
    st.markdown("---")

    st.markdown('<div class="sidebar-header">Data file</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload train.txt", type=["txt"])

    use_demo = False
    data_path = None

    if uploaded:
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        tmp.write(uploaded.read())
        tmp.close()
        data_path = tmp.name
    else:
        st.info("Upload your **train.txt** to train the model, or use the demo mode below.")
        use_demo = st.checkbox("Use demo mode (mock predictions)", value=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-header">About</div>', unsafe_allow_html=True)
    st.markdown("""
    **Model**: Logistic Regression  
    **Features**: TF-IDF vectorizer  
    **Classes**: 6 emotions  
    **Pipeline**: Text cleaning → Vectorization → Classification
    """)

    st.markdown("---")
    st.markdown('<div class="sidebar-header">Example inputs</div>', unsafe_allow_html=True)
    examples = {
        "😢 Sadness": "I feel so alone and nobody understands me",
        "😠 Anger": "I cannot believe they betrayed me like that",
        "❤️ Love": "I am so grateful to have you in my life",
        "😲 Surprise": "I had absolutely no idea this was going to happen",
        "😨 Fear": "I am terrified about what is coming next",
        "😄 Joy": "Today was the best day of my entire life",
    }
    for label, ex in examples.items():
        if st.button(label, key=label):
            st.session_state["input_text"] = ex

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧠 Emotion Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">NLP · TF-IDF + Logistic Regression · 6 emotion classes</div>',
    unsafe_allow_html=True
)

# Model status
if data_path:
    with st.spinner("Training model on your data..."):
        tfidf, model, acc, n_samples, dist = load_and_train(data_path)
        stop_words = download_nltk()
    model_ready = True

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{acc*100:.1f}%</div><div class="metric-lbl">Test accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{n_samples:,}</div><div class="metric-lbl">Training samples</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><div class="metric-val">6</div><div class="metric-lbl">Emotion classes</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

elif use_demo:
    model_ready = False
    stop_words  = download_nltk()
    st.markdown(
        '<p style="text-align:center;color:#f59e0b;font-size:0.9rem;">⚡ Demo mode — upload train.txt in the sidebar for real predictions</p>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
else:
    st.warning("Please upload your **train.txt** file in the sidebar to get started.")
    st.stop()

# ── Input ─────────────────────────────────────────────────────────────────────
input_text = st.text_area(
    "Enter your text below",
    value=st.session_state.get("input_text", ""),
    height=140,
    placeholder="Type something like: I feel incredibly happy today...",
    label_visibility="collapsed",
    key="text_input"
)

predict_clicked = st.button("🔍  Predict Emotion", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_clicked and input_text.strip():

    with st.spinner("Analyzing..."):

        if model_ready:
            emotion_name, scores = predict_emotion(input_text, tfidf, model, stop_words)
        else:
            # Demo mode — rule-based mock
            txt_lower = input_text.lower()
            if any(w in txt_lower for w in ["happy", "joy", "great", "wonderful", "best", "love it"]):
                emotion_name = "joy"
            elif any(w in txt_lower for w in ["sad", "alone", "miss", "cry", "lonely", "depressed"]):
                emotion_name = "sadness"
            elif any(w in txt_lower for w in ["angry", "furious", "rage", "hate", "betray", "cannot believe"]):
                emotion_name = "anger"
            elif any(w in txt_lower for w in ["love", "adore", "cherish", "grateful", "darling"]):
                emotion_name = "love"
            elif any(w in txt_lower for w in ["scared", "terrified", "fear", "afraid", "terrifying"]):
                emotion_name = "fear"
            else:
                emotion_name = "surprise"

            # Build dummy scores
            rng = np.random.default_rng(abs(hash(input_text)) % 2**32)
            raw = rng.dirichlet(np.ones(6) * 0.5)
            # Boost predicted class
            idx = list(LABEL_MAP.values()).index(emotion_name)
            raw[idx] += 1.0
            raw /= raw.sum()
            scores = {LABEL_MAP[i]: float(raw[i]) for i in range(6)}

    emo = EMOTIONS[emotion_name]
    conf = scores[emotion_name] * 100

    # ── Result card ──
    st.markdown(f"""
    <div class="emotion-card" style="background:{emo['bg']};border:1.5px solid {emo['color']}40;">
        <span class="emotion-emoji">{emo['emoji']}</span>
        <div class="emotion-name" style="color:{emo['color']};">{emotion_name.capitalize()}</div>
        <div class="emotion-conf">Confidence: <strong>{conf:.1f}%</strong></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability bar chart ──
    st.markdown("#### Emotion probabilities")

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    labels  = [e.capitalize() for e, _ in sorted_scores]
    values  = [v * 100 for _, v in sorted_scores]
    colors  = [EMOTIONS[e]["color"] for e, _ in sorted_scores]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.55, zorder=3)

    # Value labels
    for bar, val in zip(bars, values[::-1]):
        ax.text(
            bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left",
            fontsize=9, color="#c4c4d4", fontweight="500"
        )

    ax.set_xlim(0, 110)
    ax.tick_params(colors="#8888aa", labelsize=10)
    ax.xaxis.label.set_color("#8888aa")
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("#2d2d4e")
    ax.tick_params(axis="x", colors="#2d2d4e", length=0)
    ax.set_xticks([])
    ax.yaxis.set_tick_params(pad=8)
    plt.yticks(color="#c4c4d4")
    plt.tight_layout(pad=1.2)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Clean text preview ──
    with st.expander("🔎 See cleaned text (after preprocessing)"):
        cleaned = clean_text(input_text, stop_words)
        st.code(cleaned if cleaned else "(empty after cleaning)", language=None)

elif predict_clicked and not input_text.strip():
    st.warning("Please enter some text before predicting.")

# ── Distribution chart (only when model is trained) ──
if data_path and model_ready:
    with st.expander("📊 Training data — emotion distribution"):
        fig2, ax2 = plt.subplots(figsize=(7, 2.8))
        fig2.patch.set_facecolor("#1a1a2e")
        ax2.set_facecolor("#1a1a2e")

        emo_names = list(dist.keys())
        emo_counts = list(dist.values())
        emo_colors = [EMOTIONS.get(e, {}).get("color", "#888") for e in emo_names]

        ax2.bar(emo_names, emo_counts, color=emo_colors, zorder=3, width=0.55)
        ax2.set_facecolor("#1a1a2e")
        ax2.tick_params(colors="#8888aa", labelsize=9)
        ax2.spines[["top", "right", "left"]].set_visible(False)
        ax2.spines["bottom"].set_color("#2d2d4e")
        ax2.set_ylabel("Samples", color="#8888aa", fontsize=9)
        ax2.yaxis.set_tick_params(colors="#8888aa")

        for i, v in enumerate(emo_counts):
            ax2.text(i, v + 50, f"{v:,}", ha="center", va="bottom",
                     fontsize=8, color="#c4c4d4")

        plt.tight_layout(pad=1.0)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)