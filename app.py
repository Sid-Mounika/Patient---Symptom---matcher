# =========================
# AI SYMPTOM CHECKER - NEAT CENTERED UI WITH BUTTON NAVIGATION
# =========================
import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


# =========================
# PAGE CONFIG (FULL SCREEN)
# =========================
st.set_page_config(
    page_title="AI Symptom Checker",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# HIDE STREAMLIT DEFAULT UI
# =========================
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
section[data-testid="stSidebar"] {display: none;}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# BACKGROUND + CENTER CARD STYLE
# =========================
def set_bg(image_url):
    st.markdown(f"""
    <style>
    .stApp {{
        background: url("{image_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }}

    .center-card {{
        background: rgba(255, 255, 255, 0.10);
        backdrop-filter: blur(20px);
        padding: 45px;
        border-radius: 20px;
        width: 500px;
        margin: auto;
        margin-top: 8vh;
        text-align: center;
        text-color: black;

        box-shadow: 0 10px 40px rgba(0,0,0,0.25);
    }}

    .title {{
        font-size: 34px;
        font-weight: bold;
        color: black;
        margin-bottom: 10px;
    }}

    .subtitle {{
        font-size: 18px;
        color: black;
        margin-bottom: 25px;
    }}

    .stButton>button {{
        width: 100%;
        border-radius: 12px;
        height: 50px;
        font-size: 16px;
        font-weight: 600;
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        color: white;
        border: none;
    }}

    .stButton>button:hover {{
        transform: scale(1.02);
        transition: 0.2s;
    }}
    </style>
    """, unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = "home"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "results" not in st.session_state:
    st.session_state.results = None
if "scores" not in st.session_state:
    st.session_state.scores = None
if "symptoms" not in st.session_state:
    st.session_state.symptoms = ""

# =========================
# LOAD MODEL & DATA
# =========================
@st.cache_resource
def load_data():
    with open("metadata.pkl", "rb") as f:
        df = pickle.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = df["combined_symptoms"].astype(str).tolist()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")
    return df, model, embeddings

df, model, embedding_matrix = load_data()

# =========================
# GROQ CLIENT (AI EXPLANATION)
# =========================
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key) if groq_api_key else None

def generate_explanation(symptoms, diseases):
    if not client:
        return "⚠️ GROQ_API_KEY not set. Add it to get AI explanation."

    prompt = f"""
    Patient Symptoms: {symptoms}
    Predicted Disease: {diseases}

    Explain in very simple medical language:
    - Why this disease matches symptoms
    - Common symptoms
    - When to consult a doctor
    Keep it short and patient-friendly.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content

# =========================
# SEARCH FUNCTION
# =========================
def search_symptoms(text):
    vector = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    sims = cosine_similarity(vector, embedding_matrix)[0]
    idx = sims.argsort()[::-1][:1]

    results = df.iloc[idx][["Disease", "combined_symptoms"]].to_dict("records")
    scores = sims[idx]
    return results, scores

# =========================
# HOME PAGE
# =========================
if st.session_state.page == "home":
    set_bg("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR-D6quZ4x__ryfgVUIw-SAk5cXXQwDoiOa6A&s")

    st.markdown("""
    <div class="center-card">
        <div class="title">🩺 AI Symptom Checker</div>
        <div class="subtitle">
        Smart Disease Prediction using AI
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🔐 Start / Login"):
            st.session_state.page = "login"
            st.rerun()

# =========================
# LOGIN PAGE
# =========================
elif st.session_state.page == "login":
    set_bg("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTULssQjkGo1W7gPM-zTnsiWSbvmKmaYqJsFQ&s")
    st.markdown("""
    <div class="center-card">
        <div class="title">🔐 Patient Login</div>
        <div class="subtitle">Enter your credentials</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "1234":
                st.session_state.logged_in = True
                st.session_state.page = "symptom"
                st.success("Login Successful")
                st.rerun()
            else:
                st.error("Invalid Username or Password")

        if st.button("⬅ Back to Home"):
            st.session_state.page = "home"
            st.rerun()

# =========================
# SYMPTOM INPUT PAGE
# =========================
elif st.session_state.page == "symptom":
    if not st.session_state.logged_in:
        st.session_state.page = "login"
        st.rerun()

    set_bg("data:image/jpeg;base64,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")

    st.markdown("""
    <div class="center-card">
        <div class="title">🔎 Enter Your Symptoms</div>
        <div class="subtitle">
        Example: fever, headache, cough
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        symptoms = st.text_area("", height=130)

        if st.button("🧬 Predict Disease"):
            if symptoms.strip():
                results, scores = search_symptoms(symptoms)
                st.session_state.results = results
                st.session_state.scores = scores
                st.session_state.symptoms = symptoms
                st.session_state.page = "result"
                st.rerun()
            else:
                st.warning("Please enter symptoms")

        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.session_state.page = "home"
            st.rerun()

# =========================
# RESULT PAGE
# =========================
elif st.session_state.page == "result":
    set_bg("https://i.fbcd.co/products/original/a-medical-illustration-background-for-worlds-health-day-with-copy-space-4-cf3c72b8b3657425f48024f3055903e4f8ac17cd3cea2666b4277f9448c846ae.jpg")

    r = st.session_state.results[0]
    score = float(st.session_state.scores[0])

    # 🔥 IMPORTANT: Convert similarity (0–1) to percentage (0–100)
    score_percent = round(score * 100, 2)

    st.markdown(f"""
    <div class="center-card">
        <div class="title">🦠 Predicted Disease</div>
        <div class="subtitle">
        <b>{r['Disease']}</b><br><br>
        Similarity Score: {score_percent}%
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    
    # =========================
    # FIXED SQUARE CHART (0–100)
    # =========================
    fig, ax = plt.subplots(figsize=(4, 4))  # Perfect square chart

    # Bar chart (0–100 scale)
    ax.bar(["Similarity"], [score_percent])

    # Fixed axis (no zoom / no stretch)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("Similarity Score (0 - 100)")

    # Smart label position (prevents overflow at 100%)
    label_y = min(score_percent + 2, 98)
    ax.text(0, label_y, f"{score_percent:.1f}%", 
            ha='center', fontweight='bold', fontsize=12)

    # Professional grid
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Center the chart perfectly
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig, use_container_width=False)

    # =========================
    # CENTERED BUTTON NAVIGATION
    # =========================
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🧠 View AI Explanation", use_container_width=True):
            st.session_state.page = "explanation"
            st.rerun()

        if st.button("🔄 Check Other Symptoms", use_container_width=True):
            st.session_state.page = "symptom"
            st.rerun()

# =========================
# EXPLANATION PAGE
# =========================
elif st.session_state.page == "explanation":
    set_bg("https://images.unsplash.com/photo-1581093458791-9d42e0d0f1c9")

    explanation = generate_explanation(
        st.session_state.symptoms,
        [r["Disease"] for r in st.session_state.results]
    )

    st.markdown(f"""
    <div class="center-card">
        <div class="title">🧠 AI Medical Explanation</div>
        <div class="subtitle">{explanation}</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("⬅ Back to Results"):
            st.session_state.page = "result"
            st.rerun()

        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.session_state.page = "home"
            st.rerun()
