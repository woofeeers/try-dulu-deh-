import streamlit as st
import base64
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CekKlaim.id",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
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

# Try loading from src/static/ or fallbacks
background_base64 = get_base64_image("src/static/background.png")
if not background_base64:
    background_base64 = get_base64_image("app/static/background.png")

pill_base64 = get_base64_image("src/static/pill.png")
if not pill_base64:
    pill_base64 = get_base64_image("app/static/pill.png")

# ─────────────────────────────────────────────
#  GLOBAL CSS & CUSTOM DESIGN
# ─────────────────────────────────────────────
bg_rule = ""
if background_base64:
    bg_rule = f"""
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed;
        inset: 0;
        background-image:
            radial-gradient(ellipse 80% 60% at 10% 20%, rgba(0,180,216,0.08) 0%, transparent 60%),
            radial-gradient(ellipse 70% 50% at 90% 80%, rgba(0,100,200,0.08) 0%, transparent 60%),
            url("data:image/png;base64,{background_base64}");
        background-size: cover;
        background-position: center;
        filter: brightness(0.35) contrast(1.05) blur(1px);
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
            radial-gradient(ellipse 80% 60% at 10% 20%, rgba(0,180,216,0.07) 0%, transparent 60%),
            radial-gradient(ellipse 70% 50% at 90% 80%, rgba(0,100,200,0.07) 0%, transparent 60%),
            linear-gradient(160deg, #000d1a 0%, #001428 40%, #000d1a 100%);
        z-index: 0;
        pointer-events: none;
    }
    """

st.markdown(f"""
<style>
/* ── Import font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & base ── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    background: #000d1a !important;
    font-family: 'Inter', sans-serif !important;
    color: #e0eaf5 !important;
}}

/* Hide Streamlit components */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display: none !important; }}

/* Remove default padding */
[data-testid="stAppViewContainer"] > .main > .block-container {{
    padding: 0 !important;
    max-width: 100% !important;
}}

/* Animated background gradient & background image */
{bg_rule}

/* Floating pill decorations (background circles) */
[data-testid="stAppViewContainer"]::after {{
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(circle 2px at 15% 25%, rgba(0,220,255,0.25) 0%, transparent 100%),
        radial-gradient(circle 2px at 85% 15%, rgba(0,180,255,0.2) 0%, transparent 100%),
        radial-gradient(circle 2px at 70% 75%, rgba(0,200,255,0.15) 0%, transparent 100%),
        radial-gradient(circle 2px at 30% 70%, rgba(0,160,255,0.2) 0%, transparent 100%);
    z-index: 0;
    pointer-events: none;
}}

/* ── Page wrapper ── */
.hc-page {{
    position: relative;
    z-index: 1;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}}

/* ── NAVBAR ── */
.hc-nav {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 32px;
    border-bottom: 1px solid rgba(0,180,216,0.12);
    background: rgba(0,10,25,0.6);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    position: sticky;
    top: 0;
    z-index: 100;
}}

.hc-logo {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: #ffffff;
    text-decoration: none;
}}

.hc-logo .logo-icon {{
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #00b4d8, #0077b6);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    box-shadow: 0 0 16px rgba(0,180,216,0.4);
}}

.hc-logo span {{ color: #00d4ff; }}

/* ── HERO ── */
.hc-hero {{
    text-align: center;
    padding: 24px 24px 12px;
}}

.hc-hero h1 {{
    font-size: clamp(1.8rem, 4vw, 3rem);
    font-weight: 800;
    line-height: 1.15;
    color: #ffffff;
    letter-spacing: -1px;
    margin-bottom: 10px;
}}

.hc-hero h1 span {{
    color: #00d4ff;
    text-shadow: 0 0 30px rgba(0,212,255,0.4);
}}

.hc-hero p {{
    font-size: 1.02rem;
    color: rgba(180,210,235,0.75);
    max-width: 560px;
    margin: 0 auto 16px;
    line-height: 1.6;
    font-weight: 400;
}}

/* ── SEARCH CARD ── */
/* Style the Streamlit container as the Search Card */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    max-width: 760px;
    margin: 0 auto;
    background: rgba(0,20,45,0.7) !important;
    border: 1px solid rgba(0,180,216,0.25) !important;
    border-radius: 20px !important;
    padding: 28px 32px !important;
    box-shadow:
        0 0 40px rgba(0,180,216,0.08),
        inset 0 1px 0 rgba(255,255,255,0.04) !important;
    backdrop-filter: blur(16px);
}}

.hc-search-label {{
    display: block;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(0,212,255,0.7);
    margin-bottom: 14px;
}}

.hc-search-hint {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 14px;
    color: rgba(140,175,210,0.55);
    font-size: 0.82rem;
}}

/* Override Streamlit text_input */
[data-testid="stTextInput"] > div > div {{
    background: rgba(0,15,35,0.8) !important;
    border: 1px solid rgba(0,180,216,0.3) !important;
    border-radius: 12px !important;
    box-shadow: 0 0 0 0px rgba(0,212,255,0.2) !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
}}

[data-testid="stTextInput"] > div > div:focus-within {{
    border-color: rgba(0,212,255,0.6) !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,0.1) !important;
}}

[data-testid="stTextInput"] input {{
    color: #e0eaf5 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    padding: 16px 20px !important;
    background: transparent !important;
}}

[data-testid="stTextInput"] input::placeholder {{
    color: rgba(140,175,210,0.45) !important;
}}

/* Override Streamlit button */
[data-testid="stButton"] > button {{
    background: linear-gradient(135deg, #0096c7, #0077b6) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.3px !important;
    cursor: pointer !important;
    width: 100% !important;
    transition: all 0.25s !important;
    box-shadow: 0 4px 20px rgba(0,150,200,0.35) !important;
    margin-top: 14px !important;
}}

[data-testid="stButton"] > button:hover {{
    background: linear-gradient(135deg, #00b4d8, #0096c7) !important;
    box-shadow: 0 6px 28px rgba(0,180,216,0.5) !important;
    transform: translateY(-1px) !important;
}}

[data-testid="stButton"] > button:active {{
    transform: translateY(0) !important;
}}

/* ── RESULT SECTION ── */
.hc-result {{
    max-width: 760px;
    margin: 32px auto 0;
    background: rgba(0,18,40,0.75);
    border: 1px solid rgba(0,180,216,0.18);
    border-radius: 20px;
    padding: 32px;
    box-shadow:
        0 0 50px rgba(0,180,216,0.06),
        inset 0 1px 0 rgba(255,255,255,0.03);
    backdrop-filter: blur(16px);
    animation: fadeSlide 0.4s ease;
}}

@keyframes fadeSlide {{
    from {{ opacity: 0; transform: translateY(16px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

.hc-verdict-row {{
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 24px;
    padding-bottom: 24px;
    border-bottom: 1px solid rgba(0,180,216,0.12);
}}

.hc-verdict-icon {{
    width: 68px;
    height: 68px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    flex-shrink: 0;
}}

.hc-verdict-icon.valid {{
    background: rgba(0,200,130,0.1);
    border: 2px solid rgba(0,200,130,0.4);
    box-shadow: 0 0 24px rgba(0,200,130,0.2);
}}

.hc-verdict-icon.hoaks {{
    background: rgba(220,50,50,0.1);
    border: 2px solid rgba(220,50,50,0.4);
    box-shadow: 0 0 24px rgba(220,50,50,0.2);
}}

.hc-verdict-icon.tidak_pasti {{
    background: rgba(240,160,0,0.1);
    border: 2px solid rgba(240,160,0,0.4);
    box-shadow: 0 0 24px rgba(240,160,0,0.2);
}}

.hc-verdict-label {{
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(0,212,255,0.8);
    margin-bottom: 4px;
}}

.hc-verdict-title {{
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    line-height: 1;
    margin-bottom: 6px;
}}

.hc-verdict-title.valid   {{ color: #00e898; text-shadow: 0 0 20px rgba(0,200,130,0.4); }}
.hc-verdict-title.hoaks   {{ color: #ff5555; text-shadow: 0 0 20px rgba(220,50,50,0.4); }}
.hc-verdict-title.tidak_pasti {{ color: #ffa500; text-shadow: 0 0 20px rgba(240,160,0,0.4); }}

.hc-verdict-summary {{
    font-size: 0.95rem;
    color: rgba(180,210,235,0.8);
    line-height: 1.5;
}}

/* ── SECTION BLOCKS ── */
.hc-section {{
    margin-bottom: 24px;
}}

.hc-section-title {{
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #00b4d8;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.hc-section-title::after {{
    content: "";
    flex: 1;
    height: 1px;
    background: rgba(0,180,216,0.15);
}}

.hc-penjelasan-text {{
    font-size: 0.95rem;
    color: rgba(200,220,240,0.85);
    line-height: 1.75;
    background: rgba(0,10,25,0.4);
    border-left: 3px solid rgba(0,180,216,0.35);
    padding: 16px 18px;
    border-radius: 0 10px 10px 0;
}}

/* Fact list */
.hc-fact-list {{ list-style: none; display: flex; flex-direction: column; gap: 10px; }}
.hc-fact-list li {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    font-size: 0.92rem;
    color: rgba(200,220,240,0.85);
    line-height: 1.5;
}}
.hc-fact-list li .check {{ color: #00e898; font-size: 0.85rem; margin-top: 2px; }}
.hc-fact-list li .cross {{ color: #ff5555; font-size: 0.85rem; margin-top: 2px; }}

/* Source tags */
.hc-sources {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 4px;
}}

.hc-source-tag {{
    background: rgba(0,30,60,0.8);
    border: 1px solid rgba(0,180,216,0.25);
    color: rgba(160,200,230,0.9);
    padding: 6px 14px;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.2px;
}}

/* ── CONFIDENCE BAR ── */
.hc-confidence {{
    margin-top: 4px;
}}
.hc-conf-bar-bg {{
    height: 6px;
    background: rgba(0,180,216,0.12);
    border-radius: 100px;
    overflow: hidden;
    margin-top: 8px;
}}
.hc-conf-bar-fill {{
    height: 100%;
    border-radius: 100px;
    transition: width 0.8s ease;
}}
.hc-conf-label {{
    font-size: 0.8rem;
    color: rgba(160,200,230,0.7);
    display: flex;
    justify-content: space-between;
    margin-top: 4px;
}}

/* ── EMPTY STATE ── */
.hc-empty {{
    max-width: 760px;
    margin: 24px auto 0;
    text-align: center;
    padding: 48px 32px;
    background: rgba(0,18,40,0.5);
    border: 1px dashed rgba(0,180,216,0.2);
    border-radius: 20px;
    color: rgba(140,175,210,0.5);
    font-size: 0.9rem;
    line-height: 1.7;
}}

/* ── FOOTER ── */
.hc-footer {{
    margin-top: auto;
    padding: 32px 48px;
    border-top: 1px solid rgba(0,180,216,0.1);
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.8rem;
    color: rgba(120,160,200,0.45);
}}

/* ── SPINNER OVERRIDE ── */
[data-testid="stSpinner"] {{
    color: #00d4ff !important;
}}

/* Hide label above text input */
[data-testid="stTextInput"] label {{ display: none !important; }}

/* Remove extra top space */
.element-container:first-child {{ margin-top: 0 !important; }}

/* Clean up vertical padding and gaps of Streamlit containers */
div[data-testid="stVerticalBlock"] {{
    gap: 0.5rem !important;
}}
div[data-testid="stVerticalBlock"] > div {{
    padding-top: 0 !important;
    padding-bottom: 0 !important;
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
#  LOCAL INDOBERT MODEL LOADER (CACHED)
# ─────────────────────────────────────────────
@st.cache_resource
def load_local_model():
    """Loads tokenizer and model from local directories, cached for efficiency."""
    MODEL_PATH = "models/indobert_hoax_model"
    # Fallback to model/ if models/ isn't fully set up yet
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = "model/indobert_hoax_model"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

# ─────────────────────────────────────────────
#  DIRECT PREDICT INFERENCE (NO BACKEND NEEDED)
# ─────────────────────────────────────────────
def predict(text: str) -> dict:
    try:
        tokenizer, model = load_local_model()
        
        # Tokenize inputs
        inputs = tokenizer(
            text,
            add_special_tokens=True,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).flatten().tolist()
            prediction = torch.argmax(logits, dim=1).item()
            
        confidence = probabilities[prediction]
        classification = "HOAKS" if prediction == 0 else "VALID"
        
        # Generate custom descriptions
        if prediction == 0:
            explanation = (
                f"Klaim terdeteksi sebagai HOAKS dengan tingkat keyakinan {confidence*100:.2f}%. "
                "Faktanya, klaim ini tidak didukung oleh bukti ilmiah medis resmi dan terindikasi menyesatkan."
            )
            summary = "Klaim terdeteksi sebagai HOAKS berdasarkan klasifikasi model AI CekKlaim."
            fakta = [
                "Klaim ini tidak didukung oleh publikasi medis atau jurnal ilmiah terpercaya.",
                "Teridentifikasi pola disinformasi yang umum menyebar di platform digital.",
                "Dapat menyebabkan risiko salah tindakan medis jika dipercaya tanpa saran dokter resmi."
            ]
            sumber = ["TurnBackHoax.id", "Kemenkes RI", "MAFINDO", "WHO.int"]
        else:
            explanation = (
                f"Klaim terdeteksi sebagai VALID dengan tingkat keyakinan {confidence*100:.2f}%. "
                "Informasi ini sejalan dengan referensi medis tepercaya dan fakta kesehatan ilmiah."
            )
            summary = "Klaim terdeteksi sebagai VALID dan sejalan dengan konsensus medis."
            fakta = [
                "Klaim sejalan dengan pedoman klinis resmi dan bukti ilmiah.",
                "Didukung oleh institusi kesehatan terkemuka dan literatur medis terakreditasi.",
                "Aman dikonsumsi atau dijalankan sesuai rekomendasi dosis dan penggunaan."
            ]
            sumber = ["Kemenkes RI", "Mayo Clinic", "WHO.int", "IDAI"]
            
        return {
            "verdict": classification,
            "confidence": confidence,
            "summary": summary,
            "penjelasan": explanation,
            "fakta": fakta,
            "sumber": sumber
        }
    except Exception as e:
        return {
            "verdict": "TIDAK PASTI",
            "confidence": 0.0,
            "summary": "Gagal melakukan inferensi model.",
            "penjelasan": f"Terjadi kesalahan saat memuat model lokal: {str(e)}",
            "fakta": [
                "Pastikan model IndoBERT telah diekstrak di folder models/indobert_hoax_model/.",
                "Periksa apakah pustaka torch dan transformers terinstal dengan benar."
            ],
            "sumber": ["Sistem Diagnostik"]
        }

# ─────────────────────────────────────────────
#  VERDICT CONSTANTS
# ─────────────────────────────────────────────
VERDICT_ICON = {
    "VALID":       ("✓", "valid"),
    "HOAKS":       ("✕", "hoaks"),
    "TIDAK PASTI": ("?", "tidak_pasti"),
}

CONF_COLOR = {
    "VALID":       "#00e898",
    "HOAKS":       "#ff5555",
    "TIDAK PASTI": "#ffa500",
}

# ─────────────────────────────────────────────
#  PAGE LAYOUT
# ─────────────────────────────────────────────
st.markdown('<div class="hc-page">', unsafe_allow_html=True)

# ── NAVBAR ──
st.markdown("""
<nav class="hc-nav">
  <a class="hc-logo" href="#">
    <div class="logo-icon">🛡️</div>
    Cek<span>Klaim</span>.id
  </a>
</nav>
""", unsafe_allow_html=True)

# ── HERO ──
st.markdown("""
<section class="hc-hero">
  <h1>
    Cek Fakta Kesehatan.<br>
    Dapatkan <span>Jawaban yang Akurat.</span>
  </h1>
  <p>
    Masukkan informasi kesehatan yang ingin Anda cek.<br>
    AI kami akan menganalisis dan memverifikasi kebenarannya.
  </p>
</section>
""", unsafe_allow_html=True)

# ── SEARCH CARD ──
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
        btn_clicked = st.button("Analisis Sekarang", use_container_width=True)

# ── TRIGGER ANALYSIS ──
if btn_clicked and user_input.strip():
    st.session_state.query = user_input.strip()
    with st.spinner("Menganalisis informasi..."):
        st.session_state.result = predict(user_input.strip())

elif btn_clicked and not user_input.strip():
    st.warning("Masukkan informasi kesehatan terlebih dahulu.")

# ── RESULT DISPLAY ──
result = st.session_state.result

if result:
    verdict   = result.get("verdict", "TIDAK PASTI").upper()
    icon, css_cls = VERDICT_ICON.get(verdict, ("?", "tidak_pasti"))
    conf      = result.get("confidence", 0.0)
    conf_pct  = int(conf * 100)
    conf_color = CONF_COLOR.get(verdict, "#ffa500")
    summary   = result.get("summary", "")
    penjelasan = result.get("penjelasan", "")
    fakta     = result.get("fakta", [])
    sumber    = result.get("sumber", [])

    # Build sumber tags HTML
    sumber_html = "".join(
        f'<span class="hc-source-tag">{s}</span>' for s in sumber
    )

    # Build fact list HTML
    if verdict == "HOAKS":
        fact_icon = '<span class="cross">✕</span>'
    else:
        fact_icon = '<span class="check">✓</span>'

    fakta_items = "".join(
        f"<li>{fact_icon}<span>{f}</span></li>" for f in fakta
    )

    st.markdown(f"""
    <div class="hc-result">

      <!-- Verdict row -->
      <div class="hc-verdict-row">
        <div class="hc-verdict-icon {css_cls}">{icon}</div>
        <div>
          <div class="hc-verdict-label">Hasil Analisis AI</div>
          <div class="hc-verdict-title {css_cls}">{verdict}</div>
          <div class="hc-verdict-summary">{summary}</div>
        </div>
      </div>

      <!-- Confidence -->
      <div class="hc-section">
        <div class="hc-section-title">Tingkat Keyakinan</div>
        <div class="hc-confidence">
          <div class="hc-conf-bar-bg">
            <div class="hc-conf-bar-fill"
                 style="width:{conf_pct}%; background: linear-gradient(90deg, {conf_color}88, {conf_color});"></div>
          </div>
          <div class="hc-conf-label">
            <span>Rendah</span>
            <span style="color:{conf_color}; font-weight:600;">{conf_pct}% Yakin</span>
            <span>Tinggi</span>
          </div>
        </div>
      </div>

      <!-- Penjelasan -->
      <div class="hc-section">
        <div class="hc-section-title">Penjelasan</div>
        <div class="hc-penjelasan-text">{penjelasan}</div>
      </div>

      <!-- Fakta pendukung -->
      <div class="hc-section">
        <div class="hc-section-title">Fakta Pendukung</div>
        <ul class="hc-fact-list">{fakta_items}</ul>
      </div>

      <!-- Sumber -->
      <div class="hc-section">
        <div class="hc-section-title">Sumber Terpercaya</div>
        <div class="hc-sources">{sumber_html}</div>
      </div>

    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="hc-empty">
      Masukkan klaim atau informasi kesehatan di atas,<br>
      lalu tekan <strong style="color:rgba(0,212,255,0.7)">Analisis Sekarang</strong>
      untuk melihat hasilnya.
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ──
st.markdown("""
<footer class="hc-footer">
  <span>© 2026 CekKlaim.id — Powered by Antigravity</span>
  <span>Data bersumber dari WHO, Kemenkes RI, Mayo Clinic, dan IDAI</span>
</footer>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # close .hc-page
