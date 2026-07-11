import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Folder Penyimpanan Terpusat di Drive E
STORAGE_DIR = "E:/GWE_3_Storage"
INPUT_FILE = os.path.join(STORAGE_DIR, "scraped_articles.csv")
DB_FILE = os.path.join(STORAGE_DIR, "vector_db.pkl")

def build_vector_db():
    print("Membaca artikel referensi medis...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File {INPUT_FILE} tidak ditemukan. Harap jalankan scraper.py terlebih dahulu.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    if df.empty:
        print("Error: Dataset artikel kosong.")
        return
        
    # Pastikan data bersih
    df['title'] = df['title'].fillna("Kesehatan")
    df['paragraph'] = df['paragraph'].fillna("")
    df['source'] = df['source'].fillna("Referensi Medis")
    
    # Kumpulkan teks untuk pembuatan index
    # Kita menggunakan gabungan Title + Paragraph agar pencarian lebih akurat secara semantik
    documents = (df['title'] + " " + df['paragraph']).tolist()
    
    print("Membangun model TF-IDF Vectorizer...")
    # Menggunakan n-gram (1, 2) untuk menangkap kombinasi frasa dengan baik
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Ubah dataframe menjadi list dict agar mudah dikueri
    articles = df.to_dict(orient='records')
    
    # Kemas semua dalam dictionary
    db_data = {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "articles": articles
    }
    
    print(f"Menyimpan database vektor ke {DB_FILE}...")
    with open(DB_FILE, 'wb') as f:
        pickle.dump(db_data, f)
        
    print("Pembuatan Database Vektor selesai dengan sukses!")
    print(f"Jumlah artikel yang berhasil diindeks: {len(articles)}")

if __name__ == "__main__":
    build_vector_db()
