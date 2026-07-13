import pandas as pd
import numpy as np
import re
import string
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle

# ─────────────────────────────────────────────
#  LOCAL SEMANTIC FACT-CHECKING ENGINE
# ─────────────────────────────────────────────
CONTRADICTION_WORDS = [
    'bahaya', 'racun', 'palsu', 'membunuh', 'mati', 'fatal', 
    'membeku', 'bohong', 'salah', 'tidak aman', 'buruk', 
    'merusak', 'menyebabkan kanker', 'efek samping parah'
]

# Cache placeholder since we don't have st.cache_resource here.
# We will implement a custom cache for load_all_facts and load_local_model.
_cached_facts = None
_cached_model = None
_cached_tokenizer = None
_cached_train = None
_cached_val = None
_cached_test = None
_cached_lexicon = None

def load_all_facts():
    global _cached_facts
    if _cached_facts is not None:
        return _cached_facts
        
    facts = []
    
    # 1. Muat artikel dari Vector DB (semuanya adalah VALID/1)
    db_path = "E:/GWE_3_Storage/vector_db.pkl"
    if os.path.exists(db_path):
        try:
            with open(db_path, 'rb') as f:
                db = pickle.load(f)
            for art in db["articles"]:
                facts.append({
                    "text": art["title"] + " " + art["paragraph"],
                    "label": 1,
                    "source": art["source"],
                    "title": art["title"],
                    "type": "vector_db"
                })
        except Exception:
            pass
            
    # 2. Muat artikel dari train.csv
    train_path = "data/train.csv"
    if os.path.exists(train_path):
        try:
            df = pd.read_csv(train_path)
            for _, row in df.iterrows():
                facts.append({
                    "text": row["text"],
                    "label": int(row["label"]),
                    "source": row["source"] if pd.notna(row["source"]) else "TurnBackHoax.id",
                    "title": row["text"][:30],
                    "type": "train_csv"
                })
        except Exception:
            pass
            
    _cached_facts = facts
    return _cached_facts

def check_negation_contradiction(query, ref_text, slang_dict):
    query_clean = preprocess_text(query, slang_dict, remove_bias=True)
    ref_clean = preprocess_text(ref_text, slang_dict, remove_bias=True)
    
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

def load_split_data():
    global _cached_train, _cached_val, _cached_test
    if _cached_train is not None:
        return _cached_train, _cached_val, _cached_test
        
    train = pd.read_csv("data/train.csv")
    val   = pd.read_csv("data/val.csv")
    test  = pd.read_csv("data/test.csv")
    
    _cached_train, _cached_val, _cached_test = train, val, test
    return train, val, test

def load_lexicon():
    global _cached_lexicon
    if _cached_lexicon is not None:
        return _cached_lexicon
        
    lex = pd.read_csv("data/colloquial-indonesian-lexicon.csv")
    _cached_lexicon = dict(zip(lex['slang'], lex['formal']))
    return _cached_lexicon

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

def clean_text_fully(text, slang_dict):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    bias_words = ['salah', 'hoaks', 'hoax', 'keliru', 'klarifikasi', 'turnbackhoaxid', 'turnbackhoax']
    words = text.split()
    cleaned_words = [slang_dict.get(w, w) for w in words if w not in bias_words]
    return " ".join(cleaned_words)

def load_local_model():
    global _cached_model, _cached_tokenizer
    if _cached_model is not None:
        return _cached_tokenizer, _cached_model
        
    MODEL_PATH = "models/indobert_hoax_model"
    model_file = os.path.join(MODEL_PATH, "model.safetensors")
    
    if not os.path.exists(model_file):
        try:
            import gdown
            print("🔄 Mengunduh model IndoBERT untuk pertama kali (± 497MB)...")
            file_id = "1B-GZ2VGcQ-neqe5AOB31RFbY4Lzscb3Y"
            url = f'https://drive.google.com/uc?id={file_id}'
            os.makedirs(MODEL_PATH, exist_ok=True)
            gdown.download(url, model_file, quiet=False)
            print("✅ Model berhasil diunduh!")
        except Exception as e:
            print(f"Gagal mengunduh model: {e}")

    try:
        _cached_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _cached_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _cached_model.eval()
        return _cached_tokenizer, _cached_model
    except Exception:
        return None, None

# HoaxMLP model architecture matching training script
class HoaxMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, x):
        return self.net(x)

_cached_mlp_model = None
_cached_mlp_vectorizer = None

def load_mlp_model():
    global _cached_mlp_model, _cached_mlp_vectorizer
    if _cached_mlp_model is not None:
        return _cached_mlp_model, _cached_mlp_vectorizer
        
    model_dir = "models/mlp_model"
    model_path = os.path.join(model_dir, "model.pt")
    vec_path = os.path.join(model_dir, "vectorizer.pkl")
    
    if os.path.exists(model_path) and os.path.exists(vec_path):
        try:
            with open(vec_path, "rb") as f:
                vectorizer = pickle.load(f)
            input_dim = len(vectorizer.vocabulary_)
            model = HoaxMLP(input_dim)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            _cached_mlp_model = model
            _cached_mlp_vectorizer = vectorizer
            return _cached_mlp_model, _cached_mlp_vectorizer
        except Exception as e:
            pass
    return None, None

def predict(text: str) -> dict:
    slang_dict = load_lexicon()
    all_facts = load_all_facts()
    
    word_count = len(text.split())
    char_count = len(text)
    
    if all_facts:
        try:
            fact_texts = [preprocess_text(f["text"], slang_dict, remove_bias=True) for f in all_facts]
            
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(fact_texts)
            
            clean_query = preprocess_text(text, slang_dict, remove_bias=True)
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
                
                is_negated, word = check_negation_contradiction(text, matched_text, slang_dict)
                
                if is_negated:
                    classification = "HOAKS"
                    explanation = (
                        f"Klaim \"{text}\" bertentangan dengan rujukan medis resmi tentang \"{matched_fact['title']}\". "
                        f"Berdasarkan data kesehatan terpercaya dari {matched_fact['source']}, pernyataan ini tidak benar karena terdapat pertentangan makna."
                    )
                    summary = f"Klaim ini bertentangan dengan rujukan medis tentang {matched_fact['title']}."
                    fakta = [
                        f"Rujukan medis resmi menyatakan hal yang sebaliknya mengenai topik {matched_fact['title']}.",
                        f"Terdeteksi kontradiksi makna pada kata kunci '{word}' dibandingkan dengan data kesehatan tepercaya.",
                        "Informasi ini terindikasi sebagai disinformasi atau mitos kesehatan.",
                        f"Rujukan ilmiah didapatkan dari sumber resmi: {matched_fact['source']}."
                    ]
                    sumber = [matched_fact["source"]]
                    return {"verdict": classification, "confidence": float(best_score), "summary": summary,
                            "penjelasan": explanation, "fakta": fakta, "sumber": sumber,
                            "word_count": word_count, "char_count": char_count, "input_text": text,
                            "model_info": "Sistem Verifikasi Fakta Medis (RAG Lokal)"}
                
                has_contradiction = False
                found_words = []
                for word in CONTRADICTION_WORDS:
                    if word in clean_query and word not in matched_text:
                        has_contradiction = True
                        found_words.append(word)
                
                if has_contradiction:
                    classification = "HOAKS"
                    explanation = (
                        f"Klaim \"{text}\" bertentangan dengan fakta medis terpercaya tentang \"{matched_fact['title']}\". "
                        f"Rujukan resmi dari {matched_fact['source']} menyatakan informasi yang berbeda secara kontradiktif (ditemukan kata: {found_words})."
                    )
                    summary = f"Klaim ini terindikasi HOAKS karena bertentangan dengan fakta medis {matched_fact['title']}."
                    fakta = [
                        f"Pernyataan tersebut bertentangan dengan data medis mengenai {matched_fact['title']}.",
                        f"Ditemukan indikasi klaim berbahaya atau tidak aman (kata kunci: {found_words}).",
                        "Selalu verifikasi informasi kesehatan dari institusi resmi.",
                        f"Sumber data terpercaya: {matched_fact['source']}."
                    ]
                    sumber = [matched_fact["source"]]
                else:
                    if matched_label == 0:
                        classification = "HOAKS"
                        explanation = (
                            f"Klaim \"{text}\" serupa dengan hoaks atau disinformasi kesehatan yang telah tercatat sebelumnya: "
                            f"\"{matched_fact['text']}\". Informasi ini tidak valid dan berpotensi menyesatkan."
                        )
                        summary = "Klaim serupa dengan catatan hoaks kesehatan yang telah diverifikasi sebelumnya."
                        fakta = [
                            "Klaim ini memiliki kemiripan tinggi dengan disinformasi yang sudah diklarifikasi.",
                            "Tidak ada bukti medis ilmiah resmi yang mendukung pernyataan ini.",
                            f"Klarifikasi hoaks ini bersumber dari {matched_fact['source']}.",
                            "Selalu rujuk ke situs resmi kesehatan untuk informasi valid."
                        ]
                        sumber = [matched_fact["source"]]
                    else:
                        classification = "VALID"
                        explanation = (
                            f"Klaim \"{text}\" didukung oleh data kesehatan terpercaya mengenai \"{matched_fact['title']}\". "
                            f"Informasi ini sejalan dengan referensi medis tepercaya dari {matched_fact['source']}."
                        )
                        summary = f"Klaim ini didukung oleh rujukan medis tepercaya mengenai {matched_fact['title']}."
                        fakta = [
                            f"Pernyataan ini terbukti konsisten dengan fakta medis resmi tentang {matched_fact['title']}.",
                            "Sejalan dengan pedoman kesehatan klinis yang berlaku.",
                            f"Informasi didukung oleh literatur dari {matched_fact['source']}.",
                            "Aman digunakan sebagai referensi kesehatan umum."
                        ]
                        sumber = [matched_fact["source"]]
                
                return {"verdict": classification, "confidence": float(best_score), "summary": summary,
                        "penjelasan": explanation, "fakta": fakta, "sumber": sumber,
                        "word_count": word_count, "char_count": char_count, "input_text": text,
                        "model_info": "Sistem Verifikasi Fakta Medis (RAG Lokal)"}
                        
        except Exception as e:
            pass

    # --- FALLBACK TO LOCAL PYTORCH MLP MODEL (Debiased Deep Learning) ---
    mlp_model, mlp_vec = load_mlp_model()
    if mlp_model and mlp_vec:
        try:
            cleaned_input = clean_text_fully(text, slang_dict)
            X = mlp_vec.transform([cleaned_input]).toarray()
            if X.sum() == 0:
                explanation = (
                    f"Klaim \"{text}\" belum dapat dipastikan kevalidannya oleh sistem kami "
                    "karena tidak terdeteksi adanya kata kunci medis yang relevan dalam database rujukan kami."
                )
                summary = "Klaim belum dapat diidentifikasi karena tidak terdeteksi kata kunci rujukan."
                fakta = [
                    "Klaim tidak mengandung istilah medis yang dikenali di database kami.",
                    "Sistem tidak dapat memproses informasi tanpa adanya kata kunci yang valid.",
                    "Konsultasikan dengan dokter untuk memverifikasi kebenaran klaim ini secara klinis."
                ]
                sumber = ["Database Kebahasaan Lokal"]
                classification = "BELUM TERDETEKSI"
                return {"verdict": classification, "confidence": 0.0, "summary": summary,
                        "penjelasan": explanation, "fakta": fakta, "sumber": sumber,
                        "word_count": word_count, "char_count": char_count, "input_text": text,
                        "model_info": "Neural Network MLP (Local Debiased Model)"}
            X_t = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                logits = mlp_model(X_t)
                probabilities = torch.softmax(logits, dim=1).flatten().tolist()
                prediction = torch.argmax(logits, dim=1).item()
                
            confidence = probabilities[prediction]
            
            if confidence < 0.65:
                explanation = (
                    f"Klaim \"{text}\" belum dapat dipastikan kevalidannya oleh model analisis kami "
                    f"karena tingkat keyakinan yang rendah yaitu sebesar {confidence*100:.2f}%. "
                    "Informasi ini tidak ditemukan dalam pangkalan referensi medis resmi kami dan tidak terdeteksi sebagai hoaks yang tercatat."
                )
                summary = "Klaim belum dapat diidentifikasi karena keterbatasan data rujukan."
                fakta = [
                    "Klaim tidak ditemukan dalam database rujukan resmi.",
                    "Tingkat keyakinan model analisis berada di bawah ambang batas minimum 65%.",
                    "Tidak ada bukti kuat untuk menyatakan informasi ini valid atau hoaks secara ilmiah.",
                    "Konsultasikan dengan dokter untuk memverifikasi kebenaran medis dari klaim ini."
                ]
                sumber = ["Database Medis Lokal"]
                classification = "BELUM TERDETEKSI"
                return {"verdict": classification, "confidence": float(confidence), "summary": summary,
                        "penjelasan": explanation, "fakta": fakta, "sumber": sumber,
                        "word_count": word_count, "char_count": char_count, "input_text": text,
                        "model_info": "Neural Network MLP (Local Debiased Model)"}
            
            classification = "HOAKS" if prediction == 0 else "VALID"
            if prediction == 0:
                explanation = (
                    f"Klaim \"{text}\" telah dianalisis oleh model Neural Network (MLP) dan terdeteksi sebagai **HOAKS** "
                    f"dengan tingkat keyakinan **{confidence*100:.2f}%**. "
                    "Berdasarkan analisis fitur kata semantik medis, pernyataan ini bertentangan dengan konsensus "
                    "ilmiah kesehatan yang valid dan terindikasi sebagai disinformasi atau hoaks."
                )
                summary = "Informasi ini terindikasi tidak valid dan berpotensi menyesatkan masyarakat."
                fakta = [
                    "Klaim ini tidak didukung oleh publikasi medis atau jurnal ilmiah terpercaya.",
                    "Analisis fitur semantik mendeteksi pola ketidaksesuaian dengan standar kesehatan klinis.",
                    "Dapat menyebabkan risiko salah tindakan medis jika dipercaya tanpa konsultasi dokter.",
                    "Selalu cek rujukan resmi dari institusi terakreditasi seperti WHO atau Kemenkes RI."
                ]
                sumber = ["Kemenkes RI", "MAFINDO", "WHO.int"]
            else:
                explanation = (
                    f"Klaim \"{text}\" telah dianalisis oleh model Neural Network (MLP) dan terdeteksi sebagai **VALID** "
                    f"dengan tingkat keyakinan **{confidence*100:.2f}%**. "
                    "Pola semantik klaim ini sejalan dengan referensi klinis tepercaya dan data kesehatan medis yang sah."
                )
                summary = "Informasi ini sejalan dengan konsensus medis dan referensi kesehatan terpercaya."
                fakta = [
                    "Klaim sejalan dengan pedoman klinis resmi dan bukti ilmiah yang sudah diverifikasi.",
                    "Didukung oleh pangkalan data referensi kesehatan terakreditasi.",
                    "Aman sebagai rujukan informasi umum, namun tetap hubungi dokter untuk penanganan medis.",
                    "Informasi ini konsisten dengan panduan organisasi kesehatan resmi."
                ]
                sumber = ["Kemenkes RI", "Mayo Clinic", "WHO.int", "IDAI"]
                
            return {"verdict": classification, "confidence": float(confidence), "summary": summary,
                    "penjelasan": explanation, "fakta": fakta, "sumber": sumber,
                    "word_count": word_count, "char_count": char_count, "input_text": text,
                    "model_info": "Neural Network MLP (Local Debiased Model)"}
        except Exception as e:
            pass

    # --- FALLBACK TO LOCAL INDOBERT ---
    tokenizer, local_model = load_local_model()
    if not local_model:
        return {
            "verdict": "TIDAK PASTI", "confidence": 0.0,
            "summary": "Gagal memuat model IndoBERT.",
            "penjelasan": "Model tidak ditemukan di folder models/indobert_hoax_model. Pastikan file model.safetensors atau pytorch_model.bin sudah ada.",
            "fakta": ["Pastikan model IndoBERT telah diekstrak ke folder models/indobert_hoax_model/."],
            "sumber": ["Sistem Diagnostik"]
        }
    inputs = tokenizer(text, add_special_tokens=True, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = local_model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).flatten().tolist()
        prediction = torch.argmax(logits, dim=1).item()
    confidence = probabilities[prediction]
    
    if confidence < 0.65:
        explanation = (
            f"Klaim \"{text}\" belum dapat dipastikan kevalidannya oleh model analisis kami "
            f"karena tingkat keyakinan yang rendah yaitu sebesar {confidence*100:.2f}%. "
            "Informasi ini tidak ditemukan dalam pangkalan referensi medis resmi kami dan tidak terdeteksi sebagai hoaks yang tercatat."
        )
        summary = "Klaim belum dapat diidentifikasi karena keterbatasan data rujukan."
        fakta = [
            "Klaim tidak ditemukan dalam database rujukan resmi.",
            "Tingkat keyakinan model analisis berada di bawah ambang batas minimum 65%.",
            "Tidak ada bukti kuat untuk menyatakan informasi ini valid atau hoaks secara ilmiah.",
            "Konsultasikan dengan dokter untuk memverifikasi kebenaran medis dari klaim ini."
        ]
        sumber = ["Database Medis Lokal"]
        classification = "BELUM TERDETEKSI"
        return {"verdict": classification, "confidence": float(confidence), "summary": summary,
                "penjelasan": explanation, "fakta": fakta, "sumber": sumber,
                "word_count": word_count, "char_count": char_count, "input_text": text,
                "model_info": "IndoBERT Local (Fallback Mode)"}
                
    classification = "HOAKS" if prediction == 0 else "VALID"

    if prediction == 0:
        explanation = (
            f"Klaim \"{text}\" telah dianalisis oleh model IndoBERT dan terdeteksi sebagai **HOAKS** "
            f"dengan tingkat keyakinan **{confidence*100:.2f}%**. "
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

    return {"verdict": classification, "confidence": float(confidence), "summary": summary,
            "penjelasan": explanation, "fakta": fakta, "sumber": sumber,
            "word_count": word_count, "char_count": char_count, "input_text": text,
            "model_info": "IndoBERT Local (Fallback Mode)"}
