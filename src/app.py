import sys
import os
# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Import core engine functions
from src.engine import (
    predict,
    load_split_data,
    load_lexicon,
    preprocess_text
)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CekKlaim.id",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #0b0f11;
        background-image: radial-gradient(circle at 50% 0%, rgba(0, 82, 255, 0.08) 0%, transparent 70%), 
                          radial-gradient(rgba(183, 196, 255, 0.05) 1px, transparent 1px);
        background-size: 100% 100%, 30px 30px;
        background-position: top center, center center;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    
    .glass-panel {
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(20px) saturate(120%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(120%) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.05), 0 8px 32px 0 rgba(0, 0, 0, 0.4) !important;
        padding: 24px !important;
        border-radius: 16px !important;
        margin-bottom: 24px !important;
        transition: all 0.3s ease !important;
    }
    
    .glass-panel:hover {
        border-color: rgba(255, 255, 255, 0.15) !important;
        background: rgba(255, 255, 255, 0.06) !important;
    }
    
    .cinematic-glow { box-shadow: 0 0 30px rgba(0, 82, 255, 0.15) !important; }
    .cyan-glow { box-shadow: 0 0 25px rgba(0, 224, 255, 0.2) !important; }
    
    .green-glow { box-shadow: 0 0 25px rgba(74, 222, 128, 0.25) !important; }
    .coral-glow { box-shadow: 0 0 25px rgba(255, 180, 171, 0.25) !important; }
    .yellow-glow { box-shadow: 0 0 25px rgba(255, 210, 138, 0.25) !important; }
    
    .highlight-text {
        color: #b7c4ff !important;
        text-shadow: 0 0 12px rgba(183, 196, 255, 0.3);
    }
    
    .secondary-text {
        color: #b9f1ff !important;
        text-shadow: 0 0 12px rgba(185, 241, 255, 0.3);
    }
    
    .accent-italic {
        font-family: 'Source Serif 4', serif !important;
        font-style: italic !important;
        color: #b9f1ff !important;
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
        border-left: 3px solid #b9f1ff;
        padding-left: 16px;
        margin: 16px 0;
    }
    
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-top: 16px;
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
    }
    .stat-card:hover {
        border-color: rgba(183, 196, 255, 0.3) !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }
    .stat-val {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        margin-bottom: 4px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    .stat-lbl {
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        color: #c3c5d9 !important;
        font-weight: 700 !important;
    }
    
    .footer-text {
        font-size: 0.8rem !important;
        color: #8d90a2 !important;
        text-align: center !important;
        padding: 24px 0 !important;
        border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
        margin-top: 48px !important;
    }

    /* ── About Page: Centered Section ── */
    .about-centered {
        text-align: center !important;
        max-width: 720px;
        margin-left: auto;
        margin-right: auto;
    }

    /* ── About Page: Hero ── */
    .about-hero-title {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #b7c4ff 0%, #00e0ff 50%, #b9f1ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 8px !important;
        line-height: 1.15 !important;
        letter-spacing: -0.03em;
    }
    .about-hero-sub {
        font-family: 'Source Serif 4', serif !important;
        font-style: italic;
        font-size: 1.1rem !important;
        color: #b9f1ff !important;
        line-height: 1.6 !important;
        margin-bottom: 40px !important;
    }

    /* ── About Page: Section Divider ── */
    .about-divider {
        display: none !important;
    }
    .about-section-title {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin-bottom: 6px !important;
        text-align: center !important;
    }
    .about-section-subtitle {
        font-size: 0.88rem !important;
        color: #8d90a2 !important;
        text-align: center !important;
        margin-bottom: 28px !important;
    }

    /* ── About Page: Story Card ── */
    .about-story-card {
        background: linear-gradient(135deg, rgba(183,196,255,0.06) 0%, rgba(0,224,255,0.03) 100%) !important;
        border: 1px solid rgba(183,196,255,0.15) !important;
        border-radius: 20px !important;
        padding: 36px 32px !important;
        text-align: center !important;
        margin-bottom: 32px !important;
        position: relative;
        overflow: hidden;
    }
    .about-story-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, #b7c4ff, #00e0ff, transparent);
    }
    .about-story-card p {
        color: #c3c5d9 !important;
        line-height: 1.75 !important;
        font-size: 0.95rem !important;
        max-width: 600px;
        margin: 0 auto !important;
    }

    /* ── About Page: Step Cards Grid ── */
    .steps-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 20px;
        margin-bottom: 32px;
    }
    @media (max-width: 640px) {
        .steps-grid { grid-template-columns: 1fr; }
    }
    .step-card {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
        padding: 28px 20px !important;
        text-align: center !important;
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1) !important;
        position: relative;
        overflow: hidden;
    }
    .step-card:hover {
        border-color: rgba(183,196,255,0.25) !important;
        background: rgba(255,255,255,0.06) !important;
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    }
    .step-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 52px; height: 52px;
        border-radius: 14px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.4rem;
        font-weight: 800;
        margin-bottom: 14px;
    }
    .step-num-1 { background: rgba(183,196,255,0.12); color: #b7c4ff; border: 1px solid rgba(183,196,255,0.25); }
    .step-num-2 { background: rgba(0,224,255,0.1); color: #00e0ff; border: 1px solid rgba(0,224,255,0.2); }
    .step-num-3 { background: rgba(185,241,255,0.1); color: #b9f1ff; border: 1px solid rgba(185,241,255,0.2); }
    .step-num-4 { background: rgba(74,222,128,0.1); color: #4ade80; border: 1px solid rgba(74,222,128,0.2); }
    .step-card h4 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin: 0 0 8px 0 !important;
    }
    .step-card p {
        color: #9da0b4 !important;
        font-size: 0.84rem !important;
        line-height: 1.55 !important;
        margin: 0 !important;
    }

    /* ── About Page: Tech Badges ── */
    .tech-badges-wrap {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 10px;
        margin-bottom: 32px;
    }
    .tech-badge {
        display: inline-block;
        background: rgba(183,196,255,0.06);
        border: 1px solid rgba(183,196,255,0.18);
        border-radius: 9999px;
        padding: 8px 22px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #b7c4ff;
        transition: all 0.3s ease;
        cursor: default;
    }
    .tech-badge:hover {
        background: rgba(183,196,255,0.14);
        border-color: rgba(183,196,255,0.4);
        transform: translateY(-1px);
    }

    /* ── About Page: Team Cards ── */
    .team-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 20px;
        max-width: 520px;
        margin: 0 auto 32px auto;
    }
    @media (max-width: 480px) {
        .team-grid { grid-template-columns: 1fr; }
    }
    .team-card {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 20px !important;
        padding: 32px 20px !important;
        text-align: center !important;
        transition: all 0.35s ease !important;
    }
    .team-card:hover {
        border-color: rgba(0,224,255,0.3) !important;
        background: rgba(255,255,255,0.06) !important;
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,82,255,0.12);
    }
    .team-avatar {
        width: 72px; height: 72px;
        border-radius: 50%;
        margin: 0 auto 14px auto;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.8rem; font-weight: 800;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .team-avatar-1 {
        background: linear-gradient(135deg, rgba(183,196,255,0.2), rgba(0,82,255,0.15));
        color: #b7c4ff;
        border: 2px solid rgba(183,196,255,0.3);
    }
    .team-avatar-2 {
        background: linear-gradient(135deg, rgba(0,224,255,0.15), rgba(185,241,255,0.1));
        color: #00e0ff;
        border: 2px solid rgba(0,224,255,0.25);
    }
    .team-name {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin-bottom: 4px !important;
    }
    .team-role {
        font-size: 0.78rem !important;
        color: #8d90a2 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }

    /* ── About Page: Quote Block ── */
    .about-quote {
        background: linear-gradient(135deg, rgba(0,224,255,0.04) 0%, rgba(183,196,255,0.04) 100%) !important;
        border: 1px solid rgba(0,224,255,0.12) !important;
        border-radius: 16px !important;
        padding: 28px 32px !important;
        text-align: center !important;
        position: relative;
        margin-bottom: 24px;
    }
    .about-quote::before {
        content: '"';
        position: absolute;
        top: 8px; left: 20px;
        font-size: 4rem;
        color: rgba(0,224,255,0.15);
        font-family: 'Source Serif 4', serif;
        line-height: 1;
    }
    .about-quote p {
        font-family: 'Source Serif 4', serif !important;
        font-style: italic;
        color: #b9f1ff !important;
        font-size: 1.05rem !important;
        line-height: 1.7 !important;
        margin: 0 !important;
    }
    .about-quote .quote-attr {
        font-family: 'Inter', sans-serif !important;
        font-style: normal;
        font-size: 0.75rem !important;
        color: #8d90a2 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        margin-top: 12px !important;
        display: block;
    }

    /* ── Result Flash Animation ── */
    @keyframes resultFlash {
        0%   { opacity: 0; transform: scale(0.96) translateY(8px); }
        50%  { opacity: 1; transform: scale(1.01) translateY(-2px); }
        100% { opacity: 1; transform: scale(1) translateY(0); }
    }
    .result-flash {
        animation: resultFlash 0.55s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    @keyframes badgeGlowPulse {
        0%   { box-shadow: 0 0 0px rgba(255,255,255,0); }
        50%  { box-shadow: 0 0 25px rgba(255,255,255,0.4); }
        100% { box-shadow: 0 0 0px rgba(255,255,255,0); }
    }
    .badge-flash {
        animation: badgeGlowPulse 1.2s ease-out forwards;
    }

    /* ── Top Navigation Bar Styles ── */
    div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    div[role="radiogroup"] {
        gap: 32px !important;
        justify-content: center;
        align-items: center;
    }
    div[role="radiogroup"] > label {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: #9da0b4 !important;
        cursor: pointer;
        padding: 4px 0 8px 0 !important;
        transition: color 0.2s ease;
        background: transparent !important;
        border: none !important;
    }
    div[role="radiogroup"] > label:hover {
        color: #ffffff !important;
    }
    div[role="radiogroup"] > label:has(input:checked) {
        color: #00e0ff !important;
        border-bottom: 2px solid #00e0ff !important;
    }
    div[role="radiogroup"] > label:has(input:checked) p {
        color: #00e0ff !important;
    }
    div[role="radiogroup"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 0.95rem !important;
        margin: 0 !important;
    }
    /* Adjust Streamlit block spacing for nav */
    div[data-testid="stRadio"] {
        margin-bottom: 0 !important;
    }

    /* ── Page Transition Animation ── */
    @keyframes pageFadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .main .block-container {
        animation: pageFadeIn 0.5s ease-out forwards;
    }

    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "query" not in st.session_state:
    st.session_state.query = ""

# ─────────────────────────────────────────────
#  PAGE 1 - HALAMAN UTAMA
# ─────────────────────────────────────────────
def page_home():
    # ── Hero Section ──
    st.markdown("""
    <div class="about-centered" style="margin-bottom: 48px; padding-top: 8px;">
        <h1 class="about-hero-title" style="font-size: 3.5rem !important;">CekKlaim.id</h1>
        <p class="about-hero-sub" style="font-size: 1.25rem !important; margin-bottom: 24px !important;">
            Portal Penjaga Kesehatan Keluarga dari Bahaya Berita Bohong
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Mengapa Penting ──
    st.markdown("""
    <div class="about-centered">
        <h2 class="about-section-title">Mengapa Kami Penting Untuk Anda</h2>
    </div>
    <div class="about-story-card">
        <p>
            Seringkali kita menerima tips kesehatan di grup percakapan keluarga seperti meminum ramuan tertentu atau menghindari obat resep dokter yang ternyata berbahaya. CekKlaim hadir sebagai asisten pribadi Anda untuk dengan mudah memeriksa kebenaran berita tersebut secara gratis. Tujuan utama kami adalah memastikan kesehatan keluarga Anda terlindung dari informasi medis palsu yang bisa membahayakan nyawa.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Cara Cek Klaim ──
    st.markdown("""
    <div class="about-centered" style="margin-top: 40px;">
        <hr class="about-divider">
        <h2 class="about-section-title">Cara Cek Berita Kesehatan</h2>
        <p class="about-section-subtitle">Tiga langkah mudah untuk mengetahui kebenaran</p>
    </div>
    <div class="steps-grid" style="grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); margin-bottom: 32px;">
        <div class="step-card">
            <div class="step-num step-num-1">1</div>
            <h4>Salin Pesan Kesehatan</h4>
            <p>Dapatkan teks atau tips kesehatan mencurigakan yang Anda terima dari pesan berantai atau media sosial lalu salin teks tersebut.</p>
        </div>
        <div class="step-card">
            <div class="step-num step-num-2">2</div>
            <h4>Tempel dan Periksa</h4>
            <p>Buka halaman Cek Klaim di aplikasi ini lalu tempelkan teks tersebut ke dalam kolom yang disediakan dan tekan tombol pencarian.</p>
        </div>
        <div class="step-card">
            <div class="step-num step-num-4">3</div>
            <h4>Dapatkan Fakta Medis</h4>
            <p>Kecerdasan buatan kami akan langsung mencocokkan pesan Anda dengan fakta ilmiah dari Kemenkes RI dan WHO secara instan.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Basis Pengetahuan ──
    st.markdown("""
    <div class="about-centered" style="margin-top: 40px;">
        <hr class="about-divider">
        <h2 class="about-section-title">Kekuatan Referensi Kami</h2>
        <p class="about-section-subtitle">Untuk menjamin keakuratan sistem didukung oleh pangkalan data tepercaya yang terus diperbarui</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        train_df, val_df, test_df = load_split_data()
        total = len(train_df) + len(val_df) + len(test_df)
        
        st.markdown(f"""
        <div class="stats-container" style="max-width: 600px; margin: 0 auto;">
            <div class="stat-card" style="border-color: rgba(185, 241, 255, 0.4);">
                <div class="stat-val" style="color: #b9f1ff;">{total:,}</div>
                <div class="stat-lbl" style="color: #b9f1ff;">Total Artikel Referensi Medis</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass

    st.markdown('<div style="margin-top: 40px;"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PAGE 2 - DASHBOARD ANALISIS
# ─────────────────────────────────────────────
def page_eda():
    # ── Hero Section ──
    st.markdown("""
    <div class="about-centered" style="margin-bottom: 48px; padding-top: 8px;">
        <h1 class="about-hero-title">Mengenal Ciri Berita Bohong</h1>
        <p class="about-hero-sub">
            Memahami perbedaan pola bahasa antara informasi kesehatan palsu dan penjelasan medis resmi.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Penjelasan Pendahuluan ──
    st.markdown("""
    <div class="about-centered">
        <hr class="about-divider">
        <h2 class="about-section-title">Mengapa Analisis Ini Penting?</h2>
    </div>
    <div class="about-story-card">
        <p>
            Sebelum sistem pintar kami dapat membedakan mana berita yang asli dan mana yang palsu, kami harus mengajarinya mengenali pola bahasa yang sering digunakan oleh penyebar hoaks. Halaman ini akan menunjukkan kepada Anda apa saja ciri-ciri tulisan hoaks berdasarkan ribuan pesan yang telah kami teliti.
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

    # ── Analisis Jumlah Berita ──
    st.markdown("""
    <div class="about-centered" style="margin-top: 40px;">
        <hr class="about-divider">
        <h2 class="about-section-title">Berita Bohong Jauh Lebih Banyak Beredar</h2>
        <p class="about-section-subtitle">Perbandingan jumlah pesan bohong dan artikel medis resmi di sekitar kita</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    counts = train_df['label'].value_counts().sort_index()

    with col1:
        st.markdown(f"""
        <div class="glass-panel" style="padding: 24px !important; border-color: rgba(255, 180, 171, 0.3); height: 100%; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 2.5rem; font-weight: 800; color: #ffb4ab; font-family: 'Plus Jakarta Sans', sans-serif;">{counts.iloc[0]:,}</div>
            <div style="font-size: 0.85rem; color: #c3c5d9; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 24px;">Contoh Pesan Hoaks</div>
            
            <div style="font-size: 2.5rem; font-weight: 800; color: #b9f1ff; font-family: 'Plus Jakarta Sans', sans-serif;">{counts.iloc[1]:,}</div>
            <div style="font-size: 0.85rem; color: #c3c5d9; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">Artikel Medis Asli</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        import altair as alt
        chart_data = pd.DataFrame({'Kelompok': ['Pesan Hoaks', 'Fakta Medis'], 'Jumlah': counts.values})
        chart = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=12, cornerRadiusTopRight=12, opacity=0.9).encode(
            x=alt.X('Kelompok:N', axis=alt.Axis(labelAngle=0, title='', labelColor='#e0e3e5', labelFontSize=12)),
            y=alt.Y('Jumlah:Q', title='', axis=alt.Axis(labelColor='#e0e3e5', titleColor='#e0e3e5', grid=True, gridColor='rgba(255,255,255,0.05)')),
            color=alt.Color('Kelompok:N', scale=alt.Scale(domain=['Pesan Hoaks', 'Fakta Medis'], range=['#ffb4ab', '#b9f1ff']), legend=None)
        ).properties(height=240).configure_view(strokeWidth=0)
        st.altair_chart(chart, use_container_width=True)

    st.markdown("""
    <div class="about-centered">
        <p style="color: #9da0b4; font-size: 0.95rem; line-height: 1.6; margin-bottom: 40px; margin-top: 16px;">
            <strong>Fakta Menarik:</strong> Grafik di atas mencerminkan kenyataan di dunia nyata. Pesan hoaks sengaja dibuat masif dan disebarkan secara agresif melampaui jumlah informasi resmi dari dokter yang sebenarnya.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Analisis Panjang Tulisan ──
    st.markdown("""
    <div class="about-centered" style="margin-top: 40px;">
        <hr class="about-divider">
        <h2 class="about-section-title">Hoaks Biasanya Ditulis Sangat Pendek</h2>
        <p class="about-section-subtitle">Membandingkan panjang kalimat antara kebohongan dan fakta</p>
    </div>
    """, unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    with col3:
        import altair as alt
        hist_data = train_df[['char_length', 'label']].copy()
        hist_data['label'] = hist_data['label'].map({0: 'Pesan Hoaks', 1: 'Fakta Medis'})
        hist = alt.Chart(hist_data).mark_area(opacity=0.6, interpolate='monotone').encode(
            x=alt.X('char_length:Q', bin=alt.Bin(maxbins=30), title='Panjang Huruf', axis=alt.Axis(labelColor='#e0e3e5', titleColor='#e0e3e5')),
            y=alt.Y('count()', title='', axis=alt.Axis(labelColor='#e0e3e5')),
            color=alt.Color('label:N', scale=alt.Scale(domain=['Pesan Hoaks', 'Fakta Medis'], range=['#ffb4ab', '#b9f1ff']), legend=alt.Legend(title='', labelColor='#ffffff', orient='top-left'))
        ).properties(height=260, title='Sebaran Panjang Tulisan').configure_view(strokeWidth=0).configure_axis(gridColor='rgba(255,255,255,0.05)')
        st.altair_chart(hist, use_container_width=True)

    with col4:
        import altair as alt
        box_data = train_df[['word_count', 'label']].copy()
        box_data['label'] = box_data['label'].map({0: 'Pesan Hoaks', 1: 'Fakta Medis'})
        box_chart = alt.Chart(box_data).mark_boxplot(size=50, extent=1.5).encode(
            x=alt.X('label:N', title='', axis=alt.Axis(labelColor='#e0e3e5', labelFontSize=12)),
            y=alt.Y('word_count:Q', title='Jumlah Kata', axis=alt.Axis(labelColor='#e0e3e5', titleColor='#e0e3e5')),
            color=alt.Color('label:N', scale=alt.Scale(domain=['Pesan Hoaks', 'Fakta Medis'], range=['#ffb4ab', '#b9f1ff']), legend=None)
        ).properties(height=260, title='Rentang Jumlah Kata').configure_view(strokeWidth=0).configure_axis(gridColor='rgba(255,255,255,0.05)')
        st.altair_chart(box_chart, use_container_width=True)

    st.markdown("""
    <div class="about-centered">
        <p style="color: #9da0b4; font-size: 0.95rem; line-height: 1.6; margin-bottom: 40px; margin-top: 16px;">
            <strong>Mengapa Demikian?</strong> Pesan bohong dibuat singkat dan langsung menakut-nakuti agar cepat dibaca dan disebarkan ulang. Sebaliknya, penjelasan medis resmi ditulis lebih panjang dan berbobot karena harus memberikan alasan ilmiah yang detail mengenai suatu penyakit.
        </p>
    </div>
    """, unsafe_allow_html=True)



# ─────────────────────────────────────────────
#  PAGE 3 - VERIFIKASI KLAIM KESEHATAN
# ─────────────────────────────────────────────
def page_prediction(threshold=0.85):
    st.markdown("""
    <div class="about-centered" style="margin-bottom: 32px; padding-top: 8px;">
        <h1 class="about-hero-title">Verifikasi Klaim Kesehatan</h1>
        <p class="about-hero-sub">Periksa apakah tips, berita, atau pesan kesehatan yang Anda terima dari WhatsApp atau media sosial adalah fakta medis atau hoaks.</p>
    </div>
    """, unsafe_allow_html=True)

    user_input = st.text_area(
        label="Tempel Pesan Kesehatan di Sini",
        placeholder="Contoh: Minum air hangat dicampur lemon secara rutin dapat mematikan seluruh virus dalam tubuh secara instan...",
        value=st.session_state.query,
        key="search_input",
        height=120
    )

    st.markdown('<p style="color: #8d90a2; font-size: 0.85rem; margin-top: -8px;">Petunjuk yaitu salin pesan lengkap dari WhatsApp atau Facebook lalu tempelkan di atas.</p>', unsafe_allow_html=True)

    btn_clicked = st.button("Mulai Analisis", use_container_width=True)

    if btn_clicked and user_input.strip():
        st.session_state.query = user_input.strip()
        with st.spinner("Mohon tunggu sebentar, sistem sedang membaca rujukan medis dari Kemenkes RI dan WHO..."):
            st.session_state.result = predict(user_input.strip(), threshold=threshold)
            st.session_state.result_ts = datetime.now().strftime('%H:%M:%S')
    elif btn_clicked and not user_input.strip():
        st.info("Tolong ketik atau tempel pesan kesehatan terlebih dahulu sebelum menekan tombol analisis.")

    result = st.session_state.result
    if result:
        verdict     = result.get("verdict", "TIDAK PASTI").upper()
        conf        = result.get("confidence", 0.0)
        conf_pct    = int(conf * 100)
        input_text  = result.get("input_text", st.session_state.query)
        penjelasan  = result.get("penjelasan", "")
        fakta       = result.get("fakta", [])
        sumber      = result.get("sumber", [])
        model_info  = result.get("model_info", "Sistem Verifikasi")

        result_ts = st.session_state.get('result_ts', '')

        st.markdown('<h2 style="margin-top: 32px; margin-bottom: 16px;">Hasil Analisis</h2>', unsafe_allow_html=True)
        
        # Verdict Badge with custom glow + flash animation (fixed glow on pill only)
        if verdict == "VALID":
            badge_html = f"""
            <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 24px;">
                <div class="result-flash badge-flash green-glow" style="background: rgba(74, 222, 128, 0.1); color: #4ade80; border: 1px solid rgba(74, 222, 128, 0.3); padding: 12px 32px; border-radius: 9999px; font-weight: 800; font-size: 1.2rem; display: flex; align-items: center; gap: 8px;">
                    INFORMASI VALID / BENAR
                </div>
            </div>
            """
            panel_border = "rgba(74, 222, 128, 0.2)"
        elif verdict == "HOAKS":
            badge_html = f"""
            <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 24px;">
                <div class="result-flash badge-flash coral-glow" style="background: rgba(255, 180, 171, 0.15); color: #ffb4ab; border: 1px solid rgba(255, 180, 171, 0.3); padding: 12px 32px; border-radius: 9999px; font-weight: 800; font-size: 1.2rem; display: flex; align-items: center; gap: 8px;">
                    HOAKS / TIDAK VALID
                </div>
            </div>
            """
            panel_border = "rgba(255, 180, 171, 0.2)"
        elif verdict in ["TIDAK DAPAT DIVERIFIKASI", "BELUM TERDETEKSI", "TIDAK PASTI"]:
            badge_html = f"""
            <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 24px;">
                <div class="result-flash badge-flash yellow-glow" style="background: rgba(255, 210, 138, 0.15); color: #ffd28a; border: 1px solid rgba(255, 210, 138, 0.3); padding: 12px 32px; border-radius: 9999px; font-weight: 800; font-size: 1.2rem; display: flex; align-items: center; gap: 8px;">
                    TIDAK DAPAT DIVERIFIKASI
                </div>
            </div>
            """
            panel_border = "rgba(255, 210, 138, 0.2)"
        else:
            badge_html = f"""
            <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 24px;">
                <div class="result-flash badge-flash yellow-glow" style="background: rgba(255, 210, 138, 0.15); color: #ffd28a; border: 1px solid rgba(255, 210, 138, 0.3); padding: 12px 32px; border-radius: 9999px; font-weight: 800; font-size: 1.2rem; display: flex; align-items: center; gap: 8px;">
                    TIDAK DAPAT DIVERIFIKASI
                </div>
            </div>
            """
            panel_border = "rgba(255, 210, 138, 0.2)"

        st.markdown(badge_html, unsafe_allow_html=True)

        # Ringkasan Analisis Medis Card (with flash animation)
        st.markdown(f"""
        <div class="glass-panel result-flash" style="border-color: {panel_border};">
            <h4 style="margin-top: 0; margin-bottom: 12px; color: #b7c4ff;">Penjelasan Medis</h4>
            <p style="color: #e0e3e5; line-height: 1.6; margin: 0;">{penjelasan}</p>
            <p style="color: #8d90a2; font-size: 0.75rem; margin-top: 16px; margin-bottom: 0;">
                Tingkat Keyakinan sebesar {conf_pct}% menggunakan {model_info}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Facts Supporting
        if fakta:
            st.markdown('<h3 style="margin-top: 24px; margin-bottom: 12px; color: #b9f1ff;">Fakta Ilmiah</h3>', unsafe_allow_html=True)
            facts_list_html = "".join(f"<li>{f}</li>" for f in fakta)
            st.markdown(f'<ul class="fact-list">{facts_list_html}</ul>', unsafe_allow_html=True)

        # Trusted Sources
        if sumber:
            st.markdown('<h3 style="margin-top: 24px; margin-bottom: 12px; color: #b9f1ff;">Rujukan Sumber Tepercaya</h3>', unsafe_allow_html=True)
            sources_html = ", ".join(sumber)
            st.markdown(f'<p style="color: #c3c5d9;">{sources_html}</p>', unsafe_allow_html=True)

        # Removed timestamps

    else:
        st.info("Hasil analisis belum keluar. Silakan masukkan pesan kesehatan pada kolom di atas, lalu klik tombol 'Mulai Analisis' untuk memeriksa kebenarannya.")

def page_about():
    # ── Hero Section ──
    st.markdown("""
    <div class="about-centered" style="margin-bottom: 48px; padding-top: 8px; position: relative; z-index: 2;">
        <h1 class="about-hero-title">Tentang CekKlaim.id</h1>
    </div>
    """, unsafe_allow_html=True)

    # ── Story Card: Mengapa Kami Membuat CekKlaim.id ──
    st.markdown("""
    <div class="about-centered" style="position: relative; z-index: 2;">
        <h2 class="about-section-title">Mengapa Kami Membuat CekKlaim.id?</h2>
        <p class="about-section-subtitle">Latar belakang yang menggerakkan kami</p>
    </div>
    <div class="about-story-card" style="position: relative; z-index: 2;">
        <p>
            Ide pembuatan CekKlaim.id bermula dari kekhawatiran sehari-hari. Kami sering melihat orang tua, kerabat, hingga teman dekat membagikan tips kesehatan yang menyesatkan di grup WhatsApp keluarga — seperti meminum cairan mentah berbahaya atau menghindari obat resep dokter. Misinformasi kesehatan seperti ini bukan sekadar berita bohong biasa, melainkan ancaman nyata bagi keselamatan jiwa. Dari situlah kami tergerak untuk melahirkan CekKlaim.id sebagai wadah verifikasi informasi medis yang andal namun sangat mudah dipahami oleh siapa saja.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Cara Kerja: 4 Step Cards Grid ──
    st.markdown("""
    <div class="about-centered" style="margin-top: 40px; position: relative; z-index: 2;">
        <h2 class="about-section-title">Cara Kerja Sistem Kami</h2>
        <p class="about-section-subtitle">Empat tahapan cerdas untuk memverifikasi kebenaran klaim kesehatan</p>
    </div>
    <div class="steps-grid" style="position: relative; z-index: 2;">
        <div class="step-card">
            <div class="step-num step-num-1">1</div>
            <h4>Normalisasi Bahasa</h4>
            <p>Sistem merapikan singkatan, ejaan tidak baku, dan bahasa gaul agar tidak ada salah paham makna kalimat sebelum dianalisis.</p>
        </div>
        <div class="step-card">
            <div class="step-num step-num-2">2</div>
            <h4>Rujukan Kemenkes &amp; WHO</h4>
            <p>Seperti asisten pintar, sistem mencari artikel referensi medis paling cocok dari database resmi WHO dan Kemenkes RI.</p>
        </div>
        <div class="step-card">
            <div class="step-num step-num-3">3</div>
            <h4>Deteksi Kontradiksi</h4>
            <p>Sistem membandingkan logika kalimat untuk mendeteksi kontradiksi makna dengan fakta ilmiah dan menandainya sebagai hoaks.</p>
        </div>
        <div class="step-card">
            <div class="step-num step-num-4">4</div>
            <h4>Prediksi Neural Network</h4>
            <p>Jika klaim sangat baru, sistem mempelajari pola bahasa untuk mengukur kemungkinan pesan tersebut hoaks atau fakta.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Teknologi yang Digunakan ──
    st.markdown("""
    <div class="about-centered" style="margin-top: 40px; position: relative; z-index: 2;">
        <h2 class="about-section-title">Teknologi yang Kami Gunakan</h2>
        <p class="about-section-subtitle">Stack teknologi di balik mesin verifikasi kami</p>
    </div>
    <div class="tech-badges-wrap" style="position: relative; z-index: 2;">
        <span class="tech-badge">Python</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">TF-IDF Vectorizer</span>
        <span class="tech-badge">MLP Neural Network</span>
        <span class="tech-badge">IndoBERT</span>
        <span class="tech-badge">RAG Pipeline</span>
        <span class="tech-badge">Scikit-learn</span>
        <span class="tech-badge">PyTorch</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Tim Kami ──
    st.markdown("""
    <div class="about-centered" style="margin-top: 40px; position: relative; z-index: 2;">
        <h2 class="about-section-title">Tim Kami</h2>
        <p class="about-section-subtitle">Orang-orang di balik CekKlaim.id</p>
    </div>
    <div class="team-grid" style="position: relative; z-index: 2;">
        <div class="team-card">
            <div class="team-avatar team-avatar-1">L</div>
            <div class="team-name">Laula</div>
            <div class="team-role">Developer & Data Scientist</div>
        </div>
        <div class="team-card">
            <div class="team-avatar team-avatar-2">N</div>
            <div class="team-name">Neta</div>
            <div class="team-role">Developer & Data Scientist</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Quote / Misi ──
    st.markdown("""
    <div class="about-quote" style="position: relative; z-index: 2;">
        <p>
            Di tengah badai lebih dari 1.500 berita bohong kesehatan yang meracuni nalar publik, CekKlaim.id hadir sebagai lentera penunjuk kebenaran medis — agar tidak ada lagi nyawa dan keluarga yang menjadi korban misinformasi.
        </p>
        <span class="quote-attr">— Misi CekKlaim.id</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  NAVIGATION & MAIN
# ─────────────────────────────────────────────
st.markdown('<style>section[data-testid="stSidebar"] {display: none !important;}</style>', unsafe_allow_html=True)

nav_col1, nav_col2 = st.columns([1, 4], vertical_alignment="center")

with nav_col1:
    st.markdown('<div style="font-family: \'Plus Jakarta Sans\', sans-serif; font-size: 1.6rem; font-weight: 800; color: #00e0ff; margin-bottom: 8px;">CekKlaim.id</div>', unsafe_allow_html=True)

with nav_col2:
    page = st.radio(
        "Navigasi Aplikasi",
        ["Halaman Utama", "Dashboard Analisis", "Cek Klaim", "Tentang Platform"],
        horizontal=True,
        label_visibility="collapsed"
    )

st.markdown('<hr style="border: none; border-top: 1px solid rgba(255,255,255,0.1); margin-top: 0; margin-bottom: 40px;">', unsafe_allow_html=True)

if page == "Halaman Utama":
    page_home()
elif page == "Dashboard Analisis":
    page_eda()
elif page == "Cek Klaim":
    page_prediction()
elif page == "Tentang Platform":
    page_about()

st.markdown('<div class="footer-text">© 2026 CekKlaim.id | GWE 2026 Data Science Challenge | Data bersumber dari WHO, Kemenkes RI, Mayo Clinic, dan IDAI</div>', unsafe_allow_html=True)
