# CekKlaim.id - AI-Driven Health Fact-Checking Platform

CekKlaim.id adalah platform berbasis web interaktif untuk memverifikasi kebenaran klaim atau informasi kesehatan berbahasa Indonesia. Sistem ini ditenagai oleh model deep learning **IndoBERT** yang telah di-fine-tune untuk membedakan antara klaim kesehatan yang valid dan hoaks.

Proyek ini diajukan untuk **GWE 2026 Data Science Challenge**.

---

## 🏆 Informasi Tim & Proyek

* **Nama Tim / Peserta:** Try Dulu Deh
* **Anggota Tim:** Arneta Alifiana, Laula Fatimatusyifa
* **Tema yang Dipilih:** Risk Prediction / Sentiment Analysis (Kesehatan)
* **Link Deployment Streamlit:** [CekKlaim.id](https://cekklaimm.streamlit.app/)
* **Model yang Digunakan:** `indobenchmark/indobert-base-p2`
* **Sumber Dataset:** Dataset Hoaks Kesehatan dari Repository Open Data Publik (TurnBackHoax, MAFINDO, dsb).

> **Pernyataan Penggunaan AI Tools:**  
> Proyek ini dikembangkan dengan bantuan AI Tools berupa **Antigravity (Gemini)** dan **GitHub Copilot** untuk pendampingan penulisan kode, perbaikan error, dan pembuatan struktur file, sesuai dengan syarat dan ketentuan yang diizinkan dalam pedoman GWE 2026.

---

## 📁 Struktur Repositori

Sesuai rekomendasi struktur repositori kompetisi:

```text
├── README.md
├── requirements.txt
├── notebooks/
│   └── analysis.ipynb        # Notebook lengkap: Preprocessing, EDA, Feature Eng, Modeling, Evaluasi, Kesimpulan
├── src/
│   ├── app.py                # Aplikasi utama Streamlit (4 halaman)
│   └── static/               # Aset gambar latar belakang dan ikon
├── data/
│   ├── train.csv             # Dataset train (stratified split)
│   ├── val.csv               # Dataset validasi
│   ├── test.csv              # Dataset test
│   ├── dataset_clean.csv     # Dataset terintegrasi hasil pembersihan
│   ├── dataset_augmented.csv # Dataset teraugmentasi kelas minoritas (Valid)
│   └── colloquial-indonesian-lexicon.csv  # Kamus slang bahasa Indonesia
└── models/
    └── indobert_hoax_model   # Bobot model IndoBERT terbaik & file konfigurasi
```

---

## 🛠️ Persiapan Lingkungan (Setup)

### 1. Prasyarat Sistem
* Python 3.10 atau 3.11
* RAM minimal 4 GB (direkomendasikan 8 GB untuk memuat model IndoBERT ke memory)

### 2. Cara Instalasi Mandiri
1. Clone repositori ini dan masuk ke direktori kerja.
2. Buat virtual environment Python:
   ```bash
   python -m venv .venv
   ```
3. Aktifkan virtual environment:
   * **Windows (PowerShell)**: `.venv\Scripts\Activate.ps1`
   * **Linux/macOS**: `source .venv/bin/activate`
4. Instal dependensi yang diperlukan:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Cara Menjalankan Aplikasi

Aplikasi Streamlit terdiri dari 4 halaman utama:
1. **Halaman Utama** — Deskripsi proyek, latar belakang, tujuan, dan ringkasan dataset.
2. **EDA Dashboard** — Visualisasi interaktif: distribusi kelas, statistik deskriptif, histogram panjang karakter, boxplot jumlah kata, pie chart proporsi, dan insight shortcut learning.
3. **Prediction / Analysis** — Halaman inti untuk memasukkan klaim dan mendapatkan klasifikasi real-time dari model IndoBERT.
4. **About / Documentation** — Penjelasan model, metrik evaluasi, cara penggunaan, informasi tim, dan pernyataan AI tools.

### Opsi A: Berjalan Secara Lokal (Local Execution)
```bash
streamlit run src/app.py
```
Buka browser di **[http://localhost:8501](http://localhost:8501)**.

### Opsi B: Berjalan di Dalam Container (Docker)
```bash
docker compose up --build
```
Aplikasi tersedia pada port **`8501`**.

---

## 📓 Notebook Analisis (`notebooks/analysis.ipynb`)

Notebook ini memuat pipeline data science **end-to-end** sesuai ketentuan GWE 2026:

| No | Tahap | Deskripsi |
|----|-------|-----------|
| 1 | **Import Pustaka** | pandas, numpy, sklearn, torch, transformers, seaborn, dll. |
| 2 | **Memuat Dataset** | Load data train/val/test + kamus slang kolokial Indonesia |
| 3 | **Data Preprocessing** | Lowercase, hapus angka/tanda baca, normalisasi slang, hapus bias label |
| 4 | **EDA** | Distribusi kelas, histogram panjang karakter, statistik deskriptif |
| 5 | **Feature Engineering** | TF-IDF Vectorizer (5000 fitur, bigram) |
| 6 | **Pemodelan ML (Baseline)** | Logistic Regression (class_weight='balanced') |
| 7 | **Evaluasi Model** | Classification report & confusion matrix untuk baseline + IndoBERT |
| 8 | **Analisis OOD / Shortcut Learning** | Pengujian klaim pendek untuk membuktikan masalah generalisasi |
| 9 | **Kesimpulan & Rekomendasi** | Ringkasan temuan dan solusi perbaikan model |

---

## 🤖 Performa Model IndoBERT
* **Akurasi**: 100% pada Dataset Pengujian
* **F1-Score**: 1.00 pada Dataset Pengujian (Stratified Split)
* **Model Base**: `indobenchmark/indobert-base-p2`

> ⚠️ **Catatan:** Performa 100% ini dipengaruhi oleh *shortcut learning* akibat bias struktural panjang teks. Lihat bagian kesimpulan di notebook untuk detail analisis dan rekomendasi.
