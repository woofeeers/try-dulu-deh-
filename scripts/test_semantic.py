import os
import pickle
import pandas as pd
import re
import string

# Path ke database vektor dan dataset latih
DB_PATH = "E:/GWE_3_Storage/vector_db.pkl"
TRAIN_PATH = "data/train.csv"

# Daftar kata yang mengindikasikan kontradiksi medis
CONTRADICTION_WORDS = [
    'bahaya', 'racun', 'palsu', 'membunuh', 'mati', 'fatal', 
    'membeku', 'bohong', 'salah', 'tidak aman', 'buruk', 
    'merusak', 'menyebabkan kanker', 'efek samping parah'
]

def load_vector_db_facts():
    facts = []
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'rb') as f:
            db = pickle.load(f)
        for art in db["articles"]:
            facts.append({
                "text": art["title"] + " " + art["paragraph"],
                "label": 1,
                "source": art["source"],
                "title": art["title"],
                "type": "vector_db"
            })
    return facts

def load_train_facts():
    facts = []
    if os.path.exists(TRAIN_PATH):
        df = pd.read_csv(TRAIN_PATH)
        for _, row in df.iterrows():
            facts.append({
                "text": row["text"],
                "label": int(row["label"]),
                "source": row["source"] if pd.notna(row["source"]) else "TurnBackHoax.id",
                "title": row["text"][:30],
                "type": "train_csv"
            })
    return facts

def preprocess(text):
    text = text.lower().strip()
    text = re.sub(r'^(salah|hoaks|keliru|klarifikasi)\b\s*', '', text)
    text = re.sub(r'^\[(salah|hoaks|keliru|klarifikasi)\]\s*', '', text)
    text = text.replace("turnbackhoax.id", "")
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    return " ".join(text.split())

def check_negation_contradiction(query, ref_text):
    query_clean = preprocess(query)
    ref_clean = preprocess(ref_text)
    
    negations = ['tidak', 'bukan', 'hoaks', 'salah', 'mitos', 'keliru', 'klarifikasi', 'tiada', 'hoax']
    keywords = ['chip', '5g', 'pelacak', 'membeku', 'beku', 'racun', 'bahaya', 'kanker', 'mati', 'magnetik', 'gps', 'kontrol']
    
    for word in keywords:
        if word in query_clean:
            ref_words = ref_clean.split()
            if word in ref_words:
                idx = ref_words.index(word)
                start = max(0, idx - 3)
                ref_context = ref_words[start:idx]
                has_ref_negation = any(neg in ref_context for neg in negations)
                
                query_words = query_clean.split()
                if word in query_words:
                    q_idx = query_words.index(word)
                    q_start = max(0, q_idx - 3)
                    q_context = query_words[q_start:q_idx]
                    has_query_negation = any(neg in q_context for neg in negations)
                    
                    if has_ref_negation and not has_query_negation:
                        return True, word
                        
    return False, None

def local_predict(query, all_facts, vectorizer):
    clean_query = preprocess(query)
    
    fact_texts = [preprocess(f["text"]) for f in all_facts]
    
    tfidf_matrix = vectorizer.fit_transform(fact_texts)
    query_vector = vectorizer.transform([clean_query])
    
    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    candidates = []
    for idx, score in enumerate(scores):
        fact = all_facts[idx]
        threshold = 0.20 if fact["type"] == "vector_db" else 0.35
        if score > threshold:
            candidates.append((score, fact))
            
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, matched_fact = candidates[0]
        
        matched_label = matched_fact["label"]
        matched_text = matched_fact["text"].lower()
        
        if matched_label == 0:
            return "HOAKS", best_score, f"Klaim ini serupa dengan hoaks yang tercatat: '{matched_fact['title']}'", matched_fact["source"]
        else:
            # Cek negasi kontradiksi terlebih dahulu
            is_negated, word = check_negation_contradiction(query, matched_text)
            if is_negated:
                return "HOAKS", best_score, f"Klaim bertentangan dengan rujukan medis resmi tentang '{matched_fact['title']}' (terdapat pernyataan negasi pada kata: {word})", matched_fact["source"]
                
            # Cek kontradiksi umum
            has_contradiction = False
            found_words = []
            for word in CONTRADICTION_WORDS:
                if word in clean_query and word not in matched_text:
                    has_contradiction = True
                    found_words.append(word)
            
            if has_contradiction:
                return "HOAKS", best_score, f"Klaim bertentangan dengan fakta medis terpercaya tentang '{matched_fact['title']}' (ditemukan kata kontradiktif: {found_words})", matched_fact["source"]
            else:
                return "VALID", best_score, f"Klaim ini didukung oleh data kesehatan terpercaya tentang '{matched_fact['title']}'", matched_fact["source"]
                
    return "INDOBERT_FALLBACK", 0.0, "Tidak ada data referensi yang cukup dekat. Menggunakan model klasifikasi.", "Model Klasifikasi"

def main():
    vector_db_facts = load_vector_db_facts()
    train_facts = load_train_facts()
    all_facts = vector_db_facts + train_facts
    print(f"Total fakta referensi terindeks: {len(all_facts)}")
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    
    test_queries = [
        "Paracetamol adalah obat penurun panas.",
        "Paracetamol penurun demam.",
        "Minum paracetamol untuk obat demam.",
        "Vaksin COVID-19 mengandung chip mikro 5G.",
        "Vaksin corona ada chip 5G nya.",
        "Vaksin covid dipasang chip pelacak.",
        "Paracetamol adalah racun berbahaya bagi tubuh.",
        "Mencuci tangan dengan sabun itu sia-sia dan berbahaya.",
        "Saya suka makan nasi goreng ayam."
    ]
    
    for q in test_queries:
        verdict, score, explanation, source = local_predict(q, all_facts, vectorizer)
        print(f"\nQuery: '{q}'")
        print(f"  -> Hasil: {verdict} (Skor Kemiripan: {score:.4f})")
        print(f"  -> Sumber Rujukan: {source}")
        print(f"  -> Penjelasan: {explanation}")

if __name__ == "__main__":
    main()
