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
        background-color: #101415;
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
    
    .cinematic-glow {
        box-shadow: 0 0 30px rgba(0, 82, 255, 0.2) !important;
    }
    .cyan-glow {
        box-shadow: 0 0 25px rgba(0, 224, 255, 0.25) !important;
    }
    .green-glow {
        box-shadow: 0 0 25px rgba(74, 222, 128, 0.2) !important;
    }
    .coral-glow {
        box-shadow: 0 0 25px rgba(255, 180, 171, 0.25) !important;
    }
    
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
    # Hero Title (No emojis, cinematic styling, simplified subtitle)
    st.markdown('<h1 class="highlight-text" style="font-size: 3rem; margin-bottom: 8px;">CekKlaim.id</h1>', unsafe_allow_html=True)
    st.markdown('<p class="secondary-text" style="font-size: 1.15rem; font-weight: 500;">Portal Penjaga Kesehatan Keluarga dari Bahaya Berita Bohong</p>', unsafe_allow_html=True)
    
    # Visi & Misi inside glass panel in layperson terms
    st.markdown("""
    <div class="glass-panel cinematic-glow">
        <h3 style="margin-top: 0; margin-bottom: 12px; color: #b7c4ff; font-size: 1.3rem;">Mengapa CekKlaim.id Penting Untuk Anda?</h3>
        <p style="color: #c3c5d9; line-height: 1.6; margin: 0; font-size: 0.95rem;">
            Seringkali kita menerima tips kesehatan di grup WhatsApp keluarga seperti meminum bahan tertentu atau menghindari obat resep dokter yang ternyata tidak benar. 
            CekKlaim.id hadir sebagai tempat bagi Anda untuk dengan mudah memeriksa kebenaran berita tersebut secara gratis, 
            memastikan kesehatan keluarga Anda terlindung dari informasi medis palsu yang bisa membahayakan nyawa.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Tujuan Proyek (Philosophical goal in a single sentence)
    st.markdown('<h2 style="margin-top: 40px; margin-bottom: 16px; font-size: 1.6rem;">Tujuan Proyek</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-panel" style="border-left: 4px solid #b7c4ff; background: rgba(183, 196, 255, 0.03) !important;">
        <p class="accent-italic" style="margin: 0; font-size: 1.15rem; border-left: none; padding-left: 0; color: #e0e3e5;">
            Di tengah badai 1.500 lebih berita bohong kesehatan yang meracuni nalar publik, CekKlaim.id hadir sebagai lentera penunjuk kebenaran medis agar tidak ada lagi nyawa dan keluarga yang menjadi korban dari bahaya misinformasi.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Cara Kerja Sederhana (3 Steps for Laypeople)
    st.markdown('<h2 style="margin-top: 40px; margin-bottom: 16px; font-size: 1.6rem;">Cara Cek Klaim</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stats-container" style="grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));">
        <div class="stat-card" style="text-align: left; padding: 20px !important;">
            <div style="font-size: 1.8rem; font-weight: 800; color: #b7c4ff; margin-bottom: 8px;">1</div>
            <h4 style="margin: 0 0 8px 0; color: #ffffff; font-size: 1.05rem;">Salin Klaim Medis</h4>
            <p style="color: #c3c5d9; font-size: 0.85rem; line-height: 1.5; margin: 0;">Dapatkan teks atau tips kesehatan mencurigakan yang Anda terima dari pesan berantai WhatsApp atau media sosial.</p>
        </div>
        <div class="stat-card" style="text-align: left; padding: 20px !important;">
            <div style="font-size: 1.8rem; font-weight: 800; color: #b9f1ff; margin-bottom: 8px;">2</div>
            <h4 style="margin: 0 0 8px 0; color: #ffffff; font-size: 1.05rem;">Tempel dan Cari</h4>
            <p style="color: #c3c5d9; font-size: 0.85rem; line-height: 1.5; margin: 0;">Buka halaman "Cek Klaim" di aplikasi ini, tempelkan teks tersebut ke dalam kolom input, lalu tekan tombol Mulai Analisis.</p>
        </div>
        <div class="stat-card" style="text-align: left; padding: 20px !important;">
            <div style="font-size: 1.8rem; font-weight: 800; color: #4ade80; margin-bottom: 8px;">3</div>
            <h4 style="margin: 0 0 8px 0; color: #ffffff; font-size: 1.05rem;">Dapatkan Fakta Medis</h4>
            <p style="color: #c3c5d9; font-size: 0.85rem; line-height: 1.5; margin: 0;">Kecerdasan buatan akan langsung memeriksa fakta ilmiah dari Kemenkes RI dan WHO untuk memberitahu Anda kebenarannya secara instan.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset Summary with custom Stats Cards (Complying with GWE requirements)
    st.markdown('<h2 style="margin-top: 40px; margin-bottom: 8px; font-size: 1.6rem;">Basis Pengetahuan Kami</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #8d90a2; margin-bottom: 24px; font-size: 0.9rem;">Untuk menjamin keakuratan verifikasi, sistem kami didukung oleh pangkalan data medis terpercaya yang terus diperbarui.</p>', unsafe_allow_html=True)

    try:
        train_df, val_df, test_df = load_split_data()
        total = len(train_df) + len(val_df) + len(test_df)
        
        st.markdown(f"""
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-val">{len(train_df):,}</div>
                <div class="stat-lbl">Data Latih</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{len(val_df):,}</div>
                <div class="stat-lbl">Data Validasi</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{len(test_df):,}</div>
                <div class="stat-lbl">Data Uji</div>
            </div>
            <div class="stat-card" style="border-color: rgba(185, 241, 255, 0.4);">
                <div class="stat-val" style="color: #b9f1ff;">{total:,}</div>
                <div class="stat-lbl" style="color: #b9f1ff;">Total Basis Data</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.info("File train.csv / val.csv / test.csv tidak ditemukan.")

    st.markdown('<div style="margin-top: 40px;"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PAGE 2 - DASHBOARD ANALISIS
# ─────────────────────────────────────────────
def page_eda():
    st.markdown('<h1 class="highlight-text">Dashboard Analisis Data</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #c3c5d9;">Memahami karakteristik berita kesehatan palsu dan valid melalui data statistik yang sederhana.</p>', unsafe_allow_html=True)

    # Arahan membaca inside a styled glass panel
    st.markdown("""
    <div class="glass-panel">
        <h4 style="margin-top: 0; margin-bottom: 8px; color: #b7c4ff;">Mengapa Analisis Data Ini Penting?</h4>
        <p style="color: #c3c5d9; line-height: 1.6; margin: 0; font-size: 0.95rem;">
            Sebelum melatih sistem pintar untuk mendeteksi berita palsu, kita harus memahami terlebih dahulu bagaimana pola bahasa yang digunakan oleh pembuat hoaks dibandingkan dengan informasi resmi dari dokter. Halaman ini menunjukkan perbedaan pola tersebut secara visual.
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

    st.markdown('<h2 style="margin-top: 32px; margin-bottom: 8px; font-size: 1.5rem;">Perbandingan Jumlah Berita</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #8d90a2; margin-bottom: 20px; font-size: 0.9rem;">Melihat perbandingan jumlah contoh berita bohong (hoaks) dan berita valid yang kami gunakan untuk melatih sistem.</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    counts = train_df['label'].value_counts().sort_index()

    with col1:
        display_counts = counts.copy()
        display_counts.index = ['HOAKS (0)', 'VALID (1)']
        st.dataframe(display_counts.rename("Jumlah"), use_container_width=True)
        ratio = counts.iloc[0] / counts.iloc[1] if counts.iloc[1] > 0 else 0
        
        st.markdown(f"""
        <div class="glass-panel" style="padding: 16px !important; border-color: rgba(255, 180, 171, 0.2); margin-top: 16px;">
            <p style="color: #ffb4ab; margin: 0; font-weight: 700; font-size: 0.85rem; line-height: 1.4;">
                Rasio Perbandingan yaitu jumlah contoh berita bohong sekitar {ratio:.1f} kali lebih banyak dibandingkan artikel valid.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        import altair as alt
        chart_data = pd.DataFrame({'Kelas': ['HOAKS (0)', 'VALID (1)'], 'Jumlah': counts.values})
        chart = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
            x=alt.X('Kelas:N', axis=alt.Axis(labelAngle=0, title='', labelColor='#e0e3e5')),
            y=alt.Y('Jumlah:Q', title='Jumlah Sampel', axis=alt.Axis(labelColor='#e0e3e5', titleColor='#e0e3e5')),
            color=alt.Color('Kelas:N', scale=alt.Scale(domain=['HOAKS (0)', 'VALID (1)'], range=['#ffb4ab', '#b9f1ff']), legend=None)
        ).properties(height=200).configure_view(strokeWidth=0).configure_axis(gridColor='rgba(255,255,255,0.05)')
        st.altair_chart(chart, use_container_width=True)

    st.markdown("""
    <p style="color: #c3c5d9; font-size: 0.9rem; line-height: 1.5; margin-top: -8px; margin-bottom: 32px;">
        <strong>Apa artinya?</strong> Bagan di atas menunjukkan bahwa database kami menampung jauh lebih banyak contoh berita palsu/hoaks (warna merah). Hal ini mencerminkan dunia nyata, di mana penyebaran kabar bohong di masyarakat (misalnya lewat grup WhatsApp) memang jauh lebih agresif dan banyak dibandingkan artikel medis resmi.
    </p>
    """, unsafe_allow_html=True)

    # Section 2: Panjang Karakter
    st.markdown('<h2 style="margin-top: 32px; margin-bottom: 8px; font-size: 1.5rem;">Panjang Tulisan Berita</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #8d90a2; margin-bottom: 20px; font-size: 0.9rem;">Statistik sederhana mengenai rata-rata jumlah huruf (karakter) dalam setiap tulisan.</p>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stats-container" style="margin-bottom: 16px;">
        <div class="stat-card">
            <div class="stat-val">{train_df['char_length'].mean():.0f}</div>
            <div class="stat-lbl">Rata-rata Karakter</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{train_df['char_length'].max():,}</div>
            <div class="stat-lbl">Tulisan Terpanjang</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{train_df['char_length'].min()}</div>
            <div class="stat-lbl">Tulisan Terpendek</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Section 3: Visual Perbandingan Karakter
    st.markdown('<h2 style="margin-top: 32px; margin-bottom: 8px; font-size: 1.5rem;">Pola Panjang Tulisan Berdasarkan Kebenaran</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #8d90a2; margin-bottom: 20px; font-size: 0.9rem;">Membandingkan visualisasi panjang kalimat antara kelompok hoaks (merah) dan valid (biru).</p>', unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    with col3:
        import altair as alt
        hist_data = train_df[['char_length', 'label']].copy()
        hist_data['label'] = hist_data['label'].map({0: 'HOAKS', 1: 'VALID'})
        hist = alt.Chart(hist_data).mark_bar(opacity=0.8).encode(
            x=alt.X('char_length:Q', bin=alt.Bin(maxbins=30), title='Panjang Huruf', axis=alt.Axis(labelColor='#e0e3e5', titleColor='#e0e3e5')),
            y=alt.Y('count()', title='Banyaknya Tulisan', axis=alt.Axis(labelColor='#e0e3e5', titleColor='#e0e3e5')),
            color=alt.Color('label:N', scale=alt.Scale(domain=['HOAKS', 'VALID'], range=['#ffb4ab', '#b9f1ff']), legend=alt.Legend(title='Kelompok', labelColor='#e0e3e5', titleColor='#e0e3e5'))
        ).properties(height=240, title='Sebaran Panjang Tulisan').configure_view(strokeWidth=0).configure_axis(gridColor='rgba(255,255,255,0.05)')
        st.altair_chart(hist, use_container_width=True)

    with col4:
        import altair as alt
        box_data = train_df[['word_count', 'label']].copy()
        box_data['label'] = box_data['label'].map({0: 'HOAKS', 1: 'VALID'})
        box_chart = alt.Chart(box_data).mark_boxplot(size=40).encode(
            x=alt.X('label:N', title='', axis=alt.Axis(labelColor='#e0e3e5')),
            y=alt.Y('word_count:Q', title='Jumlah Kata', axis=alt.Axis(labelColor='#e0e3e5', titleColor='#e0e3e5')),
            color=alt.Color('label:N', scale=alt.Scale(domain=['HOAKS', 'VALID'], range=['#ffb4ab', '#b9f1ff']), legend=None)
        ).properties(height=240, title='Rentang Jumlah Kata Per Kelompok').configure_view(strokeWidth=0).configure_axis(gridColor='rgba(255,255,255,0.05)')
        st.altair_chart(box_chart, use_container_width=True)

    st.markdown("""
    <p style="color: #c3c5d9; font-size: 0.9rem; line-height: 1.5; margin-top: 12px; margin-bottom: 32px;">
        <strong>Apa artinya?</strong> Kedua grafik di atas membuktikan bahwa pesan berita bohong/hoaks (warna merah) hampir seluruhnya ditulis dengan sangat singkat, padat, dan langsung menyerang (seperti pesan berantai WhatsApp). Sebaliknya, penjelasan medis resmi yang valid (warna biru) ditulis dengan kalimat yang panjang, detail, dan ilmiah oleh para ahli medis untuk menjelaskan diagnosis secara akurat.
    </p>
    """, unsafe_allow_html=True)

    st.markdown('<h2 style="margin-top: 32px; margin-bottom: 8px; font-size: 1.5rem;">Proporsi Pembagian Data</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #8d90a2; margin-bottom: 20px; font-size: 0.9rem;">Diagram lingkaran yang menunjukkan porsi pembagian data untuk melatih sistem pintar kami.</p>', unsafe_allow_html=True)
    
    col5, col6 = st.columns(2)

    with col5:
        import altair as alt
        pie_data = pd.DataFrame({'Kelas': ['HOAKS', 'VALID'], 'Jumlah': counts.values})
        pie_chart = alt.Chart(pie_data).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Jumlah", type="quantitative"),
            color=alt.Color(field="Kelas", type="nominal", scale=alt.Scale(domain=['HOAKS', 'VALID'], range=['#ffb4ab', '#b9f1ff']), title='Kelas', legend=alt.Legend(labelColor='#e0e3e5', titleColor='#e0e3e5')),
            tooltip=['Kelas', 'Jumlah']
        ).properties(title='Proporsi Kelas Data Latih', height=240).configure_view(strokeWidth=0)
        st.altair_chart(pie_chart, use_container_width=True)

    with col6:
        try:
            import altair as alt
            sizes = [len(train_df), len(val_df), len(test_df)]
            split_data = pd.DataFrame({
                'Dataset': [f'Latih ({sizes[0]:,})', f'Validasi ({sizes[1]:,})', f'Uji ({sizes[2]:,})'],
                'Jumlah': sizes
            })
            split_chart = alt.Chart(split_data).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Jumlah", type="quantitative"),
                color=alt.Color(field="Dataset", type="nominal", scale=alt.Scale(range=['#b9f1ff', '#b7c4ff', '#c2c6db']), title='Dataset', legend=alt.Legend(labelColor='#e0e3e5', titleColor='#e0e3e5')),
                tooltip=['Dataset', 'Jumlah']
            ).properties(title='Pembagian Dataset Utama', height=240).configure_view(strokeWidth=0)
            st.altair_chart(split_chart, use_container_width=True)
        except Exception:
            pass

    st.markdown("""
    <div class="glass-panel" style="border-color: rgba(255, 180, 171, 0.25); margin-top: 32px;">
        <h4 style="color: #ffb4ab; margin-top: 0; margin-bottom: 8px;">Mengapa Kami Menggabungkan Model Klasifikasi dan Pencarian Referensi Lokal?</h4>
        <p style="color: #c3c5d9; line-height: 1.6; margin: 0; font-size: 0.95rem;">
            <strong>Tantangan Utama yaitu</strong> Karena sebagian besar hoaks ditulis pendek dan sebagian besar artikel valid ditulis panjang, model biasa akan dengan mudah 'terkecoh' karena model tersebut akan langsung menebak semua tulisan pendek sebagai hoaks tanpa membaca isinya terlebih dahulu. 
            <br><br>
            <strong>Solusi Kami yaitu</strong> Untuk mengatasi kelemahan model tersebut, sistem kami menggabungkan pencarian artikel referensi. Sistem kami benar-benar membaca kandungan isi kueri Anda dan mencocokkannya dengan database medis resmi Kemenkes RI dan WHO sebelum mengambil keputusan verifikasi fakta secara akurat.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PAGE 3 - VERIFIKASI KLAIM KESEHATAN
# ─────────────────────────────────────────────
def page_prediction():
    st.markdown('<h1 class="highlight-text">Verifikasi Klaim Kesehatan</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #c3c5d9;">Periksa apakah tips, berita, atau pesan kesehatan yang Anda terima dari WhatsApp atau media sosial adalah fakta medis atau hoaks.</p>', unsafe_allow_html=True)

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
            st.session_state.result = predict(user_input.strip())
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

        st.markdown('<h2 style="margin-top: 32px; margin-bottom: 16px;">Hasil Analisis</h2>', unsafe_allow_html=True)
        
        # Verdict Badge with custom glow
        if verdict == "VALID":
            badge_html = f"""
            <div style="display: flex; justify-content: center; margin-bottom: 24px;">
                <div class="green-glow" style="background: rgba(74, 222, 128, 0.1); color: #4ade80; border: 1px solid rgba(74, 222, 128, 0.3); padding: 12px 32px; border-radius: 9999px; font-weight: 800; font-size: 1.2rem; display: flex; align-items: center; gap: 8px;">
                    INFORMASI VALID / BENAR
                </div>
            </div>
            """
            panel_border = "rgba(74, 222, 128, 0.2)"
        elif verdict == "HOAKS":
            badge_html = f"""
            <div style="display: flex; justify-content: center; margin-bottom: 24px;">
                <div class="coral-glow" style="background: rgba(255, 180, 171, 0.15); color: #ffb4ab; border: 1px solid rgba(255, 180, 171, 0.3); padding: 12px 32px; border-radius: 9999px; font-weight: 800; font-size: 1.2rem; display: flex; align-items: center; gap: 8px;">
                    HOAKS / TIDAK VALID
                </div>
            </div>
            """
            panel_border = "rgba(255, 180, 171, 0.2)"
        else:
            badge_html = f"""
            <div style="display: flex; justify-content: center; margin-bottom: 24px;">
                <div style="background: rgba(255, 210, 138, 0.1); color: #ffd28a; border: 1px solid rgba(255, 210, 138, 0.2); padding: 12px 32px; border-radius: 9999px; font-weight: 800; font-size: 1.2rem; display: flex; align-items: center; gap: 8px;">
                    TIDAK PASTI ({conf_pct}%)
                </div>
            </div>
            """
            panel_border = "rgba(255, 210, 138, 0.2)"

        st.markdown(badge_html, unsafe_allow_html=True)

        # Ringkasan Analisis Medis Card
        st.markdown(f"""
        <div class="glass-panel" style="border-color: {panel_border};">
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

        now = datetime.now().strftime("%d %B %Y, %H.%M WIB")
        st.caption(f"Analisis selesai pada {now}")

    else:
        st.info("Hasil analisis belum keluar. Silakan masukkan pesan kesehatan pada kolom di atas, lalu klik tombol 'Mulai Analisis' untuk memeriksa kebenarannya.")

def page_about():
    st.markdown('<h1 class="highlight-text">Tentang Platform CekKlaim.id</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #c3c5d9;">Kisah perjuangan kami di balik layar dan cara kerja sistem pintar kami untuk melindungi keluarga Anda.</p>', unsafe_allow_html=True)

    # 1. Kisah di Balik Layar (Why)
    st.markdown("""
    <div class="glass-panel">
        <h4 style="margin-top: 0; margin-bottom: 12px; color: #b7c4ff; font-size: 1.25rem;">Mengapa Kami Membuat CekKlaim.id?</h4>
        <p style="color: #c3c5d9; line-height: 1.7; margin: 0; font-size: 0.95rem;">
            Ide pembuatan CekKlaim.id bermula dari kekhawatiran setiap hari. Kami sering melihat orang tua, kerabat, hingga teman dekat membagikan tips kesehatan yang menyesatkan di grup WhatsApp keluarga seperti meminum cairan mentah berbahaya atau menghindari obat resep dokter. Misinformasi kesehatan seperti ini bukan sekadar berita bohong biasa, melainkan ancaman nyata bagi keselamatan jiwa. Dari situlah kami tergerak untuk melahirkan CekKlaim.id sebagai wadah verifikasi informasi medis yang andal namun sangat mudah dipahami oleh siapa saja, bahkan bagi orang awam sekalipun.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 2. Bagaimana AI Kami Bekerja (How)
    st.markdown("""
    <div class="glass-panel">
        <h4 style="margin-top: 0; margin-bottom: 16px; color: #b7c4ff; font-size: 1.25rem;">Bagaimana Cara Sistem Memeriksa Kebenaran Pesan Anda?</h4>
        <p style="color: #c3c5d9; line-height: 1.6; margin-bottom: 20px; font-size: 0.95rem;">
            Agar mendapatkan hasil analisis dengan tingkat akurasi tinggi layaknya seorang dokter, sistem pintar kami bekerja dengan empat tahapan terstruktur berikut
        </p>
        <div style="display: flex; flex-direction: column; gap: 20px;">
            <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); padding: 16px; border-radius: 12px;">
                <strong style="color: #b9f1ff; font-size: 1rem; display: block; margin-bottom: 6px;">1. Menormalisasi Bahasa Chat</strong>
                <span style="color: #c3c5d9; font-size: 0.88rem; line-height: 1.5; display: block;">
                    Sebelum dianalisis, sistem merapikan singkatan, ejaan tidak baku, dan bahasa gaul yang sering digunakan di aplikasi percakapan sehari hari agar tidak ada salah paham makna kalimat.
                </span>
            </div>
            <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); padding: 16px; border-radius: 12px;">
                <strong style="color: #b9f1ff; font-size: 1rem; display: block; margin-bottom: 6px;">2. Membuka Rujukan Resmi Kemenkes & WHO (RAG)</strong>
                <span style="color: #c3c5d9; font-size: 0.88rem; line-height: 1.5; display: block;">
                    Seperti asisten pintar, sistem kami langsung mencari artikel referensi medis paling cocok dari database resmi WHO dan Kemenkes RI yang telah kami kumpulkan untuk mencocokkan fakta ilmiah terbaru.
                </span>
            </div>
            <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); padding: 16px; border-radius: 12px;">
                <strong style="color: #b9f1ff; font-size: 1rem; display: block; margin-bottom: 6px;">3. Uji Kebohongan dan Logika Negasi</strong>
                <span style="color: #c3c5d9; font-size: 0.88rem; line-height: 1.5; display: block;">
                    Sistem membandingkan logika kalimat. Jika referensi ilmiah menyatakan suatu bahan tidak menyebabkan pembekuan darah, sedangkan pesan yang Anda tempel tertulis menyebabkan pembekuan darah, sistem akan mendeteksi kontradiksi makna ini dan segera menandainya sebagai HOAKS.
                </span>
            </div>
            <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); padding: 16px; border-radius: 12px;">
                <strong style="color: #b9f1ff; font-size: 1rem; display: block; margin-bottom: 6px;">4. Prediksi Pintar (Klasifikasi Neural Network)</strong>
                <span style="color: #c3c5d9; font-size: 0.88rem; line-height: 1.5; display: block;">
                    Jika klaim kesehatan yang Anda tanyakan sangat baru dan belum tercatat di database medis resmi, sistem pintar kami akan mempelajari pola struktur bahasanya untuk mengukur seberapa besar kemungkinan pesan tersebut bernada hoaks atau fakta.
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  NAVIGATION & MAIN
# ─────────────────────────────────────────────
st.sidebar.markdown('<h2 style="color: #b7c4ff; font-size: 1.5rem; margin-top: 0; margin-bottom: 24px; font-weight: 800; font-family: \'Plus Jakarta Sans\', sans-serif;">CekKlaim.id</h2>', unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigasi Aplikasi",
    ["Halaman Utama", "Dashboard Analisis", "Cek Klaim", "Tentang Platform"]
)

if page == "Halaman Utama":
    page_home()
elif page == "Dashboard Analisis":
    page_eda()
elif page == "Cek Klaim":
    page_prediction()
elif page == "Tentang Platform":
    page_about()

st.markdown('<div class="footer-text">© 2026 CekKlaim.id | GWE 2026 Data Science Challenge | Data bersumber dari WHO, Kemenkes RI, Mayo Clinic, dan IDAI</div>', unsafe_allow_html=True)
