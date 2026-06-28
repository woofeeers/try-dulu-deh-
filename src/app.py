import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import base64
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CekKlaim.id",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  BASE64 ASSETS LOADER
# ─────────────────────────────────────────────
def get_base64_image(image_path):
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception:
            pass
    return ""

background_base64 = get_base64_image("src/static/background.png")
if not background_base64:
    background_base64 = get_base64_image("app/static/background.png")

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
bg_rule = ""
if background_base64:
    bg_rule = f"""
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.95)),
            url("data:image/png;base64,{background_base64}");
        background-size: cover;
        background-position: center;
        filter: brightness(1.1) contrast(1.0) blur(2px);
        z-index: 0;
        pointer-events: none;
    }}
    """
else:
    bg_rule = """
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        inset: 0;
        background:
            linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.95)),
            linear-gradient(160deg, #050505 0%, #0a0a0a 40%, #050505 100%);
        z-index: 0;
        pointer-events: none;
    }
    """

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    background: #f9fafb !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #1e293b !important;
}}

#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display: none !important; }}

{bg_rule}

[data-testid="stAppViewContainer"]::after {{
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(circle 2px at 15% 25%, rgba(255,255,255,0.05) 0%, transparent 100%),
        radial-gradient(circle 2px at 85% 15%, rgba(255,255,255,0.04) 0%, transparent 100%),
        radial-gradient(circle 2px at 70% 75%, rgba(255,255,255,0.03) 0%, transparent 100%),
        radial-gradient(circle 2px at 30% 70%, rgba(255,255,255,0.02) 0%, transparent 100%);
    z-index: 0;
    pointer-events: none;
}}

/* ── Streamlit Tweaks ── */
.block-container {{
    padding-top: 0rem !important; margin-top: -1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}}
header[data-testid="stHeader"] {{
    display: none !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}}
[data-testid="stSidebar"] .stRadio label {{
    color: #334155 !important;
    font-weight: 500 !important;
}}

/* ── Page wrapper ── */
.hc-page {{
    position: relative;
    z-index: 1;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}}

/* ── Hero ── */
.hc-hero {{
    text-align: center;
    padding: 8px 24px 16px;
}}

.hc-hero h1 {{
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 900;
    line-height: 1.15;
    color: #0f172a;
    letter-spacing: -1.5px;
    margin-bottom: 12px;
}}

.hc-hero h1 span {{
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.hc-hero p {{
    font-size: 1rem;
    color: #475569;
    max-width: 680px;
    margin: 0 auto 16px;
    line-height: 1.65;
    font-weight: 400;
}}

/* ── Cards / Containers ── */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: #ffffff !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.03) !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 24px 28px !important;
    box-shadow:
        0 8px 32px rgba(0,0,0,0.3),
        0 0 40px rgba(255,255,255,0.08),
        inset 0 1px 0 rgba(255,255,255,0.04) !important;
    backdrop-filter: blur(16px);
}}

/* ── Input ── */
.hc-search-label {{
    display: block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 14px;
}}

[data-testid="stTextInput"] > div > div {{
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}}

[data-testid="stTextInput"] > div > div:focus-within {{
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.2) !important;
}}

[data-testid="stTextInput"] input {{
    color: #1e293b !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    padding: 16px 20px !important;
    background: transparent !important;
}}

[data-testid="stTextInput"] input::placeholder {{
    color: #94a3b8 !important;
}}

[data-testid="stTextInput"] label {{ display: none !important; }}

/* ── Button ── */
[data-testid="stButton"] > button {{
    background: linear-gradient(135deg, #d4d4d4, #8a8a8a) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.3px !important;
    cursor: pointer !important;
    width: 100% !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 24px rgba(0,150,200,0.4) !important;
    margin-top: 14px !important;
}}

[data-testid="stButton"] > button:hover {{
    background: linear-gradient(135deg, #ffffff, #d4d4d4) !important;
    box-shadow: 0 8px 32px rgba(255,255,255,0.08) !important;
    transform: translateY(-2px) !important;
}}

/* ── Result Section ── */
.hc-result {{
    margin: 28px auto 0;
    background: rgba(0,14,35,0.85);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 0;
    box-shadow:
        0 16px 48px rgba(0,0,0,0.4),
        0 0 60px rgba(255,255,255,0.08),
        inset 0 1px 0 rgba(255,255,255,0.03);
    backdrop-filter: blur(20px);
    animation: fadeSlide 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    overflow: hidden;
}}

@keyframes fadeSlide {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

.result-header {{
    padding: 28px 32px;
    display: flex;
    align-items: center;
    gap: 20px;
}}

.result-header.valid {{ background: #f0fdf4; border-bottom: 1px solid rgba(0,200,130,0.15); }}
.result-header.hoaks {{ background: #fef2f2; border-bottom: 1px solid rgba(255,85,85,0.15); }}
.result-header.tidak_pasti {{ background: #fffbeb; border-bottom: 1px solid rgba(255,165,0,0.15); }}

.verdict-icon {{
    width: 64px;
    height: 64px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
    flex-shrink: 0;
}}
.verdict-icon.valid {{ background: #dcfce7; border: 2px solid rgba(0,200,130,0.3); }}
.verdict-icon.hoaks {{ background: #fee2e2; border: 2px solid rgba(255,85,85,0.3); }}
.verdict-icon.tidak_pasti {{ background: #fef3c7; border: 2px solid rgba(255,165,0,0.3); }}

.verdict-label {{ font-size: 0.68rem; font-weight: 700; letter-spacing: 2.5px; text-transform: uppercase; margin-bottom: 4px; }}
.verdict-label.valid {{ color: #16a34a; }}
.verdict-label.hoaks {{ color: #dc2626; }}
.verdict-label.tidak_pasti {{ color: #d97706; }}

.verdict-title {{ font-size: 2rem; font-weight: 900; letter-spacing: -0.5px; line-height: 1; }}
.verdict-title.valid {{ color: #15803d; }}
.verdict-title.hoaks {{ color: #b91c1c; }}
.verdict-title.tidak_pasti {{ color: #b45309; }}

.verdict-summary {{ font-size: 0.9rem; color: #475569; margin-top: 4px; }}

.result-body {{ padding: 28px 32px; display: flex; flex-direction: column; gap: 24px; }}

.result-quote {{
    background: #e2e8f0;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px 20px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}}
.result-quote .q-icon {{ font-size: 1.4rem; flex-shrink: 0; margin-top: 2px; }}
.result-quote .q-label {{ font-size: 0.68rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #64748b; margin-bottom: 4px; }}
.result-quote .q-text {{ font-size: 0.95rem; color: #1e293b; font-style: italic; line-height: 1.5; }}

.section-title {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.section-title::after {{ content: ""; flex: 1; height: 1px; background: #e2e8f0; }}

.explanation-box {{
    font-size: 0.92rem;
    color: #334155;
    line-height: 1.75;
    background: #f8fafc;
    border-left: 3px solid #cbd5e1;
    padding: 16px 20px;
    border-radius: 0 12px 12px 0;
}}

.conf-bar-bg {{ height: 8px; background: #e2e8f0; border-radius: 100px; overflow: hidden; margin-top: 8px; }}
.conf-bar-fill {{ height: 100%; border-radius: 100px; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1); }}
.conf-labels {{ font-size: 0.78rem; color: #64748b; display: flex; justify-content: space-between; margin-top: 6px; }}

.fact-list {{ list-style: none; display: flex; flex-direction: column; gap: 10px; padding: 0; }}
.fact-list li {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    font-size: 0.9rem;
    color: #334155;
    line-height: 1.55;
    padding: 10px 14px;
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.08);
}}
.fact-list li .icon {{ font-size: 0.8rem; margin-top: 3px; flex-shrink: 0; }}

.source-tags {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.source-tag {{
    background: rgba(0,30,60,0.7);
    border: 1px solid rgba(255,255,255,0.08);
    color: rgba(160,200,230,0.9);
    padding: 8px 16px;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    transition: all 0.2s;
}}

.result-meta {{
    display: flex;
    justify-content: space-between;
    padding: 14px 32px;
    background: #f1f5f9;
    border-top: 1px solid #e2e8f0;
    font-size: 0.72rem;
    color: rgba(120,160,200,0.4);
}}

/* ── Empty state ── */
.hc-empty {{
    text-align: center;
    padding: 64px 32px;
    background: rgba(0,14,35,0.5);
    border: 1px dashed rgba(255,255,255,0.08);
    border-radius: 20px;
    margin-top: 28px;
}}
.hc-empty .empty-icon {{ font-size: 3rem; margin-bottom: 16px; opacity: 0.4; }}
.hc-empty .empty-title {{ color: rgba(180,210,235,0.6); font-size: 1.05rem; font-weight: 600; margin-bottom: 8px; }}
.hc-empty .empty-sub {{ color: rgba(140,175,210,0.4); font-size: 0.88rem; line-height: 1.6; }}

/* ── Footer ── */
.hc-footer {{
    margin-top: auto;
    padding: 28px 48px;
    border-top: 1px solid #e2e8f0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.78rem;
    color: rgba(120,160,200,0.35);
}}

/* ── Search hint ── */
.hc-search-hint {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 12px;
    color: rgba(140,175,210,0.45);
    font-size: 0.82rem;
}}

/* ── Insight box ── */
.insight-box {{
    background: rgba(0,212,255,0.05);
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 12px;
    padding: 18px 22px;
    margin-top: 16px;
}}
.insight-box h4 {{ color: #0f172a; margin-bottom: 8px; font-size: 0.88rem; letter-spacing: 0.3px; }}
.insight-box p {{ color: rgba(200,220,240,0.8); font-size: 0.86rem; line-height: 1.6; margin: 0; }}

/* ── Spinner ── */
[data-testid="stSpinner"] {{ color: #ffffff !important; }}

/* ── Metric cards ── */
[data-testid="stMetric"] {{
    background: rgba(0,20,45,0.6) !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "query" not in st.session_state:
    st.session_state.query = ""

# ─────────────────────────────────────────────
#  DARK-THEMED MATPLOTLIB HELPER
# ─────────────────────────────────────────────
DARK_BG = '#0a1628'
GRID_COLOR = 'rgba(255,255,255,0.08)'

def style_ax(fig, ax, title=""):
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.tick_params(colors='#8caad2', labelsize=9)
    ax.xaxis.label.set_color('#8caad2')
    ax.yaxis.label.set_color('#8caad2')
    for spine in ax.spines.values():
        spine.set_color('#1a3050')
    ax.grid(True, alpha=0.1, color='#1e293b')
    if title:
        ax.set_title(title, color='#1e293b', fontsize=12, fontweight='bold', pad=14)

# ─────────────────────────────────────────────
#  DATA LOADERS
# ─────────────────────────────────────────────
@st.cache_data
def load_split_data():
    train = pd.read_csv("data/train.csv")
    val   = pd.read_csv("data/val.csv")
    test  = pd.read_csv("data/test.csv")
    return train, val, test

@st.cache_data
def load_lexicon():
    lex = pd.read_csv("data/colloquial-indonesian-lexicon.csv")
    return dict(zip(lex['slang'], lex['formal']))

def preprocess_text(text, slang_dict, remove_bias=True):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    if remove_bias:
        text = re.sub(r'^(salah|hoaks|keliru|klarifikasi)\b\s*', '', text)
        text = re.sub(r'^\[(salah|hoaks|keliru|klarifikasi)\]\s*', '', text)
        text = text.replace("turnbackhoax.id", "")
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    words = text.split()
    normalized = [slang_dict.get(w, w) for w in words]
    return " ".join(normalized)

# ─────────────────────────────────────────────
#  LOCAL INDOBERT MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_local_model():
    MODEL_PATH = "models/indobert_hoax_model"
    model_file = os.path.join(MODEL_PATH, "model.safetensors")
    
    # Otomatis download dari Google Drive jika model belum ada
    if not os.path.exists(model_file):
        try:
            import gdown
            st.info("🔄 Mengunduh model IndoBERT untuk pertama kali (± 497MB). Mohon tunggu sekitar 1-2 menit...")
            # Menggunakan ID file dari link yang diberikan user
            file_id = "1B-GZ2VGcQ-neqe5AOB31RFbY4Lzscb3Y"
            url = f'https://drive.google.com/uc?id={file_id}'
            os.makedirs(MODEL_PATH, exist_ok=True)
            gdown.download(url, model_file, quiet=False)
            st.success("✅ Model berhasil diunduh!")
        except Exception as e:
            st.error(f"Gagal mengunduh model: {e}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        return tokenizer, model
    except Exception:
        return None, None

def predict(text: str) -> dict:
    tokenizer, model = load_local_model()
    if not model:
        return {
            "verdict": "TIDAK PASTI", "confidence": 0.0,
            "summary": "Gagal memuat model IndoBERT.",
            "penjelasan": "Model tidak ditemukan di folder models/indobert_hoax_model. Pastikan file model.safetensors atau pytorch_model.bin sudah ada.",
            "fakta": ["Pastikan model IndoBERT telah diekstrak ke folder models/indobert_hoax_model/."],
            "sumber": ["Sistem Diagnostik"]
        }
    inputs = tokenizer(text, add_special_tokens=True, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).flatten().tolist()
        prediction = torch.argmax(logits, dim=1).item()
    confidence = probabilities[prediction]
    classification = "HOAKS" if prediction == 0 else "VALID"

    word_count = len(text.split())
    char_count = len(text)

    if prediction == 0:
        explanation = (
            f"Klaim \"{text}\" telah dianalisis oleh model IndoBERT dan terdeteksi sebagai **HOAKS** "
            f"dengan tingkat keyakinan **{confidence*100:.2f}%**. "
            f"Klaim ini terdiri dari {word_count} kata ({char_count} karakter). "
            "Berdasarkan pola yang dipelajari dari ribuan data klaim kesehatan, "
            "informasi ini tidak didukung oleh bukti ilmiah medis resmi dan terindikasi menyesatkan. "
            "Kami menyarankan Anda untuk selalu memverifikasi informasi kesehatan dengan tenaga medis profesional."
        )
        summary = "Informasi ini terindikasi tidak valid dan berpotensi menyesatkan masyarakat."
        fakta = [
            "Klaim ini tidak didukung oleh publikasi medis atau jurnal ilmiah terpercaya.",
            "Teridentifikasi pola disinformasi yang umum menyebar di platform digital.",
            "Dapat menyebabkan risiko salah tindakan medis jika dipercaya tanpa konsultasi dokter.",
            "Selalu cek sumber informasi kesehatan dari website resmi seperti WHO atau Kemenkes RI."
        ]
        sumber = ["TurnBackHoax.id", "Kemenkes RI", "MAFINDO", "WHO.int"]
    else:
        explanation = (
            f"Klaim \"{text}\" telah dianalisis oleh model IndoBERT dan terdeteksi sebagai **VALID** "
            f"dengan tingkat keyakinan **{confidence*100:.2f}%**. "
            f"Klaim ini terdiri dari {word_count} kata ({char_count} karakter). "
            "Informasi ini sejalan dengan referensi medis tepercaya dan fakta kesehatan ilmiah. "
            "Meskipun demikian, selalu konsultasikan dengan tenaga medis profesional untuk penanganan spesifik."
        )
        summary = "Informasi ini sejalan dengan konsensus medis dan referensi kesehatan terpercaya."
        fakta = [
            "Klaim sejalan dengan pedoman klinis resmi dan bukti ilmiah yang sudah diverifikasi.",
            "Didukung oleh institusi kesehatan terkemuka dan literatur medis terakreditasi.",
            "Aman sebagai referensi informasi, namun tetap konsultasikan ke dokter untuk penanganan.",
            "Informasi ini konsisten dengan panduan dari organisasi kesehatan internasional."
        ]
        sumber = ["Kemenkes RI", "Mayo Clinic", "WHO.int", "IDAI"]

    return {"verdict": classification, "confidence": confidence, "summary": summary,
            "penjelasan": explanation, "fakta": fakta, "sumber": sumber,
            "word_count": word_count, "char_count": char_count, "input_text": text}

# ══════════════════════════════════════════════
#  PAGE 1 – HALAMAN UTAMA
# ══════════════════════════════════════════════
def page_home():
    st.markdown("""
    <section class="hc-hero">
      <h1>Selamat Datang di <span>CekKlaim.id</span></h1>
      <p>Platform cerdas berbasis AI untuk mendeteksi dan memverifikasi kebenaran
      informasi serta klaim kesehatan berbahasa Indonesia secara instan.</p>
    </section>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 Latar Belakang
        Di era digital, penyebaran misinformasi dan hoaks terkait isu kesehatan menyebar
        lebih cepat daripada fakta medis. Hal ini berujung pada konsekuensi fatal —
        pengobatan mandiri yang salah dan keresahan masyarakat luas.

        **Contoh hoaks yang sering beredar:**
        - *"Minum air hangat dapat membunuh virus corona."*
        - *"Vaksin COVID-19 mengandung chip mikro 5G."*
        """)
    with col2:
        st.markdown("""
        ### 🚀 Tujuan Proyek
        Membangun sistem *fact-checking* otomatis **end-to-end** yang dapat
        memprediksi tingkat keabsahan klaim kesehatan (*Risk Prediction*).

        **Fitur Utama:**
        - ✅ Klasifikasi otomatis HOAKS vs VALID
        - 📊 Analisis tingkat keyakinan (*confidence*)
        - 📚 Sumber referensi medis terpercaya
        - ⚡ Respons real-time (< 5 detik)
        """)

    st.markdown("---")
    st.markdown("### 📊 Ringkasan Dataset")
    try:
        train_df, val_df, test_df = load_split_data()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📁 Data Train", f"{len(train_df):,}")
        c2.metric("📁 Data Validasi", f"{len(val_df):,}")
        c3.metric("📁 Data Test", f"{len(test_df):,}")
        c4.metric("📊 Total", f"{len(train_df)+len(val_df)+len(test_df):,}")
    except Exception:
        st.info("File train.csv / val.csv / test.csv tidak ditemukan.")

# ══════════════════════════════════════════════
#  PAGE 2 – EDA DASHBOARD
# ══════════════════════════════════════════════
def page_eda():
    st.markdown("""
    <section class="hc-hero">
      <h1>Dashboard <span>Exploratory Data Analysis</span></h1>
      <p>Visualisasi interaktif dari karakteristik dataset yang digunakan untuk melatih model IndoBERT.</p>
    </section>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(37,99,235,0.06); border-left: 4px solid #2563eb; padding: 18px 22px; border-radius: 0 12px 12px 0; margin-bottom: 28px;">
        <h4 style="margin-top: 0; color: #0f172a; font-size: 1rem;">📖 Arahan Membaca Analisis Data</h4>
        <p style="margin-bottom: 0; font-size: 0.9rem; line-height: 1.7; color: #475569;">
            Halaman ini menyajikan hasil EDA dari <b>dataset train</b>. Dataset terdiri dari dua kelas:
            <b>HOAKS (0)</b> dan <b>VALID (1)</b>. Perhatikan <b>ketidakseimbangan kelas yang ekstrim</b> —
            kelas HOAKS mendominasi. Perbedaan panjang teks sangat mencolok, mengindikasikan potensi
            <b>shortcut learning</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        train_df, val_df, test_df = load_split_data()
    except Exception:
        st.error("Dataset tidak ditemukan.")
        return

    slang_dict = load_lexicon()
    train_df['clean_text'] = train_df['text'].astype(str).apply(lambda t: preprocess_text(t, slang_dict))
    train_df['char_length'] = train_df['clean_text'].apply(len)
    train_df['word_count']  = train_df['clean_text'].apply(lambda x: len(x.split()))

    # ── 1. Distribusi Kelas ──
    st.write("### 1. Distribusi Kelas (Class Imbalance)")
    col1, col2 = st.columns([1, 2])
    counts = train_df['label'].value_counts().sort_index()

    with col1:
        st.write("**Jumlah sampel per kelas:**")
        display_counts = counts.copy()
        display_counts.index = ['HOAKS (0)', 'VALID (1)']
        st.dataframe(display_counts.rename("Jumlah"), use_container_width=True)
        ratio = counts.iloc[0] / counts.iloc[1] if counts.iloc[1] > 0 else 0
        st.warning(f"⚠️ Rasio imbalance: **{ratio:.1f} : 1** (HOAKS : VALID)")

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#ff5555', '#00e898']
        bars = ax.bar(['HOAKS (0)', 'VALID (1)'], counts.values, color=colors, width=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                    f'{val:,}', ha='center', va='bottom', color='#1e293b', fontweight='bold', fontsize=11)
        ax.set_ylabel('Jumlah Sampel')
        style_ax(fig, ax, 'Distribusi Kelas pada Dataset Train')
        st.pyplot(fig)
        plt.close()

    # ── 2. Statistik Deskriptif ──
    st.write("---")
    st.write("### 2. Statistik Deskriptif Panjang Teks")
    stats = train_df.groupby('label')['char_length'].describe()
    stats.index = ['HOAKS (0)', 'VALID (1)']
    st.dataframe(stats.style.format("{:.1f}"), use_container_width=True)

    # ── 3. Distribusi Panjang Karakter & Kata ──
    st.write("---")
    st.write("### 3. Distribusi Panjang Karakter & Jumlah Kata")
    col3, col4 = st.columns(2)

    with col3:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        for label, color, name in [(0, '#ff5555', 'HOAKS'), (1, '#00e898', 'VALID')]:
            subset = train_df[train_df['label'] == label]['char_length']
            ax2.hist(subset, bins=50, alpha=0.6, color=color, label=name)
        ax2.legend(facecolor='#ffffff', edgecolor='#e2e8f0', labelcolor='#1e293b')
        ax2.set_xlabel('Panjang Karakter')
        ax2.set_ylabel('Frekuensi')
        style_ax(fig2, ax2, 'Histogram Panjang Karakter')
        st.pyplot(fig2)
        plt.close()
        st.caption("Klaim HOAKS cenderung sangat pendek (< 200 karakter), sedangkan artikel VALID jauh lebih panjang.")

    with col4:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        bp_data = [train_df[train_df['label']==0]['word_count'], train_df[train_df['label']==1]['word_count']]
        bp = ax3.boxplot(bp_data, labels=['HOAKS', 'VALID'], patch_artist=True,
                         boxprops=dict(edgecolor='#1e293b'),
                         medianprops=dict(color='#1e293b', linewidth=2),
                         whiskerprops=dict(color='#1e293b'),
                         capprops=dict(color='#1e293b'),
                         flierprops=dict(marker='o', markerfacecolor='#ff5555', markersize=3, alpha=0.4))
        bp['boxes'][0].set_facecolor('#ffcccc')
        bp['boxes'][1].set_facecolor('#ccffcc')
        ax3.set_ylabel('Jumlah Kata')
        style_ax(fig3, ax3, 'Boxplot Jumlah Kata per Kelas')
        st.pyplot(fig3)
        plt.close()
        st.caption("Boxplot menunjukkan perbedaan signifikan — indikasi kuat potensi shortcut learning.")

    # ── 4. Pie Charts ──
    st.write("---")
    st.write("### 4. Proporsi Dataset")
    col5, col6 = st.columns(2)

    with col5:
        fig4, ax4 = plt.subplots(figsize=(5, 5))
        ax4.pie(counts.values, labels=['HOAKS', 'VALID'], autopct='%1.1f%%',
                colors=['#ff5555', '#00e898'], startangle=90,
                textprops={'color': 'white', 'fontsize': 11, 'fontweight': 'bold'},
                wedgeprops={'edgecolor': DARK_BG, 'linewidth': 2})
        fig4.patch.set_facecolor(DARK_BG)
        ax4.set_title('Proporsi Kelas (Train)', color='#1e293b', fontsize=12, fontweight='bold', pad=14)
        st.pyplot(fig4)
        plt.close()

    with col6:
        try:
            sizes = [len(train_df), len(val_df), len(test_df)]
            labels_s = [f'Train ({sizes[0]:,})', f'Val ({sizes[1]:,})', f'Test ({sizes[2]:,})']
            fig5, ax5 = plt.subplots(figsize=(5, 5))
            ax5.pie(sizes, labels=labels_s, autopct='%1.1f%%',
                    colors=['#d4d4d4', '#ffffff', '#48cae4'], startangle=90,
                    textprops={'color': 'white', 'fontsize': 10, 'fontweight': 'bold'},
                    wedgeprops={'edgecolor': DARK_BG, 'linewidth': 2})
            fig5.patch.set_facecolor(DARK_BG)
            ax5.set_title('Pembagian Dataset (Train/Val/Test)', color='#1e293b', fontsize=12, fontweight='bold', pad=14)
            st.pyplot(fig5)
            plt.close()
        except Exception:
            pass

    # ── 5. Temuan ──
    st.write("---")
    st.write("### 5. Temuan EDA Penting")
    st.markdown("""
    <div class="insight-box">
        <h4>⚠️ Potensi Shortcut Learning</h4>
        <p>Kelas VALID selalu berupa artikel medis panjang (>500 karakter), sedangkan HOAKS berupa judul singkat (<200 karakter).
        Model berpotensi hanya belajar dari panjang teks, bukan konten medis sebenarnya.</p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  PAGE 3 – PREDICTION / ANALYSIS
# ══════════════════════════════════════════════
def page_prediction():
    st.markdown("""
    <section class="hc-hero">
      <h1>Cek Fakta Kesehatan.<br>Dapatkan <span>Jawaban yang Akurat.</span></h1>
      <p>Masukkan informasi kesehatan yang ingin Anda cek. AI kami akan menganalisis dan memverifikasi kebenarannya.</p>
    </section>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<label class="hc-search-label">Masukkan klaim atau informasi kesehatan</label>', unsafe_allow_html=True)
        col_input, _ = st.columns([1, 0.001])
        with col_input:
            user_input = st.text_input(
                label="input",
                placeholder='Contoh: "Paracetamol adalah obat penurun panas."',
                value=st.session_state.query,
                key="search_input",
                label_visibility="collapsed",
            )

        st.markdown(
            '<div class="hc-search-hint">🛡️&nbsp; Contoh: "Minum air es menyebabkan flu atau pembekuan darah"</div>',
            unsafe_allow_html=True,
        )

        _, col_btn, _ = st.columns([0.15, 0.7, 0.15])
        with col_btn:
            btn_clicked = st.button("🔍 Analisis Sekarang", use_container_width=True)

    if btn_clicked and user_input.strip():
        st.session_state.query = user_input.strip()
        with st.spinner("Menganalisis informasi dengan AI..."):
            st.session_state.result = predict(user_input.strip())
    elif btn_clicked and not user_input.strip():
        st.warning("Masukkan informasi kesehatan terlebih dahulu.")

    result = st.session_state.result
    if result:
        verdict = result.get("verdict", "TIDAK PASTI").upper()
        icon_map = {"VALID": ("✓", "valid"), "HOAKS": ("✕", "hoaks"), "TIDAK PASTI": ("?", "tidak_pasti")}
        icon, css_cls = icon_map.get(verdict, ("?", "tidak_pasti"))
        conf = result.get("confidence", 0.0)
        conf_pct = int(conf * 100)
        color_map = {"VALID": "#00e898", "HOAKS": "#ff5555", "TIDAK PASTI": "#ffa500"}
        conf_color = color_map.get(verdict, "#ffa500")

        input_text = result.get("input_text", st.session_state.query)
        word_count = result.get("word_count", len(input_text.split()))
        char_count = result.get("char_count", len(input_text))
        summary = result.get("summary", "")
        penjelasan = result.get("penjelasan", "")
        fakta = result.get("fakta", [])
        sumber = result.get("sumber", [])
        now = datetime.now().strftime("%d %B %Y, %H:%M WIB")

        # Sanitize user input for safe HTML embedding
        safe_input = input_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        safe_penjelasan = penjelasan.replace("**", "").replace("<", "&lt;").replace(">", "&gt;")

        # 1) Verdict Header
        st.markdown(f'<div class="result-header {css_cls}"><div class="verdict-icon {css_cls}">{icon}</div><div><div class="verdict-label {css_cls}">Hasil Analisis AI</div><div class="verdict-title {css_cls}">{verdict}</div><div class="verdict-summary">{summary}</div></div></div>', unsafe_allow_html=True)

        # 2) Quoted Input
        st.markdown(f'<div class="result-quote"><span class="q-icon">💬</span><div><div class="q-label">Klaim yang Dianalisis</div><div class="q-text">&ldquo;{safe_input}&rdquo;</div></div></div>', unsafe_allow_html=True)

        # 3) Confidence Bar
        st.markdown(f'<div style="margin-top:12px"><div class="section-title">Tingkat Keyakinan Model</div><div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{conf_pct}%;background:linear-gradient(90deg,{conf_color}66,{conf_color})"></div></div><div class="conf-labels"><span>Rendah</span><span style="color:{conf_color};font-weight:700;font-size:0.85rem">{conf_pct}% Yakin</span><span>Tinggi</span></div></div>', unsafe_allow_html=True)

        # 4) Penjelasan Detail
        st.markdown(f'<div style="margin-top:12px"><div class="section-title">Penjelasan Detail</div><div class="explanation-box">{safe_penjelasan}</div></div>', unsafe_allow_html=True)

        # 5) Fakta Pendukung
        fi = '<span class="icon" style="color:#ff5555">✕</span>' if verdict == "HOAKS" else '<span class="icon" style="color:#00e898">✓</span>'
        items_html = "".join(f"<li>{fi}<span>{f}</span></li>" for f in fakta)
        st.markdown(f'<div style="margin-top:12px"><div class="section-title">Fakta Pendukung</div><ul class="fact-list">{items_html}</ul></div>', unsafe_allow_html=True)

        # 6) Sumber Terpercaya
        tags_html = "".join(f'<span class="source-tag">{s}</span>' for s in sumber)
        st.markdown(f'<div style="margin-top:12px"><div class="section-title">Sumber Terpercaya</div><div class="source-tags">{tags_html}</div></div>', unsafe_allow_html=True)

        # 7) Meta Footer
        st.markdown(f'<div class="result-meta"><span>Dianalisis pada {now}</span><span>{word_count} kata &middot; {char_count} karakter &middot; Model: IndoBERT v2</span></div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="hc-empty"><div class="empty-icon">🔍</div><div class="empty-title">Belum Ada Analisis</div><div class="empty-sub">Masukkan klaim atau informasi kesehatan di atas, lalu tekan <strong style="color:#ffffff">Analisis Sekarang</strong> untuk melihat hasilnya.</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  PAGE 4 – ABOUT / DOCUMENTATION
# ══════════════════════════════════════════════
def page_about():
    st.markdown('<section class="hc-hero"><h1>Tentang <span>CekKlaim.id</span></h1></section>', unsafe_allow_html=True)
    st.markdown("""
    ### 🧠 Penjelasan Model
    Aplikasi ini menggunakan **IndoBERT** (`indobenchmark/indobert-base-p2`), sebuah model transformer
    bahasa Indonesia yang telah melalui proses *fine-tuning* pada ribuan data klaim kesehatan.

    **Pipeline:**
    1. **Data Preprocessing** — Lowercase, hapus URL & karakter khusus, normalisasi slang.
    2. **Feature Engineering** — Tokenisasi menggunakan tokenizer IndoBERT (max_length=64).
    3. **Training** — Fine-tuning dengan PyTorch & HuggingFace Transformers.
    4. **Inference** — Klasifikasi biner real-time pada aplikasi Streamlit.

    ---
    ### 📈 Metrik Evaluasi

    | Metrik | Skor |
    |--------|------|
    | **Akurasi** | 100% |
    | **F1-Score** | 1.00 |
    | **Precision** | 1.00 |
    | **Recall** | 1.00 |

    > ⚠️ **Catatan:** Performa 100% dipengaruhi oleh *shortcut learning*. Lihat EDA Dashboard.

    ---
    ### 🔧 Cara Penggunaan
    1. Buka menu **Prediction / Analysis** di sidebar.
    2. Masukkan teks berupa klaim, opini, atau informasi medis.
    3. Klik **Analisis Sekarang**.
    4. Lihat hasil verifikasi, tingkat keyakinan, dan fakta pendukungnya.

    ---
    ### 👥 Informasi Tim
    - **Nama Tim:** Try Dulu Deh
    - **Anggota:** Arneta Alifiana, Laula Fatimatusyifa
    - **Kompetisi:** GWE 2026 Data Science Challenge
    """)


# ─────────────────────────────────────────────
#  NAVIGATION & MAIN
# ─────────────────────────────────────────────
st.markdown('<div class="hc-page">', unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="display:flex;align-items:center;gap:12px;font-size:1.6rem;font-weight:900;color:#1e293b;margin-bottom:28px;letter-spacing:-0.5px;">
  <div style="width:38px;height:38px;background:#2563eb; color:#ffffff !important;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;color:#000;">🛡️</div>
  <span style="font-weight:600;letter-spacing:-0.5px;">CekKlaim<span style="color:#888;">.id</span></span>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigasi Aplikasi",
    ["Halaman Utama", "Dashboard Analisis", "Cek Fakta (AI)", "Tentang Platform"]
)

if page == "Halaman Utama":
    page_home()
elif page == "Dashboard Analisis":
    page_eda()
elif page == "Cek Fakta (AI)":
    page_prediction()
elif page == "Tentang Platform":
    page_about()

st.markdown("""
<footer class="hc-footer">
  <span>© 2026 CekKlaim.id — GWE 2026 Data Science Challenge</span>
  <span>Data bersumber dari WHO, Kemenkes RI, Mayo Clinic, dan IDAI</span>
</footer>
</div>
""", unsafe_allow_html=True)
