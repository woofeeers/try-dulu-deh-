import os
import pickle

DB_PATH = "E:/GWE_3_Storage/vector_db.pkl"

def main():
    print("=== PENGUJIAN LOKAL RETRIEVAL RAG ===")
    if not os.path.exists(DB_PATH):
        print(f"Error: Database vektor {DB_PATH} tidak ditemukan.")
        return
        
    print(f"Memuat database vektor dari {DB_PATH}...")
    with open(DB_PATH, 'rb') as f:
        db = pickle.load(f)
        
    vectorizer = db["vectorizer"]
    tfidf_matrix = db["tfidf_matrix"]
    articles = db["articles"]
    
    # Query pengujian OOD
    query = "Paracetamol adalah obat penurun panas."
    print(f"\nQuery input: '{query}'")
    
    # Vectorize query
    query_vector = vectorizer.transform([query])
    
    # Hitung cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Cari top-3 indeks tertinggi
    top_k = 3
    top_indices = scores.argsort()[-top_k:][::-1]
    
    print("\nArtikel medis yang berhasil dicocokkan (Top-3):")
    for i, idx in enumerate(top_indices, 1):
        art = articles[idx]
        score = scores[idx]
        print(f"{i}. [{art['title']}] (Skor Kemiripan: {score:.4f})")
        print(f"   Konten: {art['paragraph'][:120]}...")
        print(f"   Sumber: {art['source']}\n")

if __name__ == "__main__":
    main()
