# CekKlaim.id - AI-Driven Health Fact-Checking Platform

CekKlaim.id adalah platform berbasis web interaktif untuk memverifikasi kebenaran klaim atau informasi kesehatan berbahasa Indonesia. Sistem ini ditenagai oleh model deep learning **IndoBERT** yang telah di-fine-tune untuk membedakan antara klaim kesehatan yang valid dan hoaks.

---

## 📁 Struktur Repositori

Sesuai rekomendasi struktur repositori, proyek ini memiliki tata letak sebagai berikut:

```text
├── README.md
├── requirements.txt
├── notebooks/
│   └── analysis.ipynb      # Notebook berisi analisis data & performa model
├── src/
│   ├── app.py              # Aplikasi utama Streamlit (In-Process ML Inference)
│   └── static/             # Aset gambar latar belakang dan ikon
├── data/
│   ├── dataset_clean.csv   # Dataset terintegrasi hasil pembersihan
│   └── dataset_augmented.csv # Dataset teraugmentasi kelas minoritas (Valid)
├── models/
│   └── indobert_hoax_model # Bobot model IndoBERT terbaik & file konfigurasi
└── presentation/
    └── slides.pdf          # Slide presentasi ringkasan proyek
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

### Opsi A: Berjalan Secara Lokal (Local Execution)
Cukup jalankan satu perintah berikut untuk memuat server Streamlit yang secara langsung melakukan klasifikasi menggunakan model IndoBERT lokal:
```bash
streamlit run src/app.py
```
Setelah berjalan, buka browser Anda dan akses halaman web di **[http://localhost:8501](http://localhost:8501)**.

### Opsi B: Berjalan di Dalam Container (Docker)
Aplikasi ini sudah mendukung containerization secara utuh. Cukup jalankan perintah berikut (pastikan aplikasi Docker Desktop Anda aktif):
```bash
docker compose up --build
```
Aplikasi akan dibundel secara otomatis dan dapat diakses langsung pada port **`8501`**.

---

## 🤖 Performa Model IndoBERT
* **Akurasi**: 100% pada Dataset Pengujian
* **F1-Score**: 1.00 pada Dataset Pengujian (Stratified Split 15%)
* **Model Base**: `indobenchmark/indobert-base-p2`
