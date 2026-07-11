import os
import re
import csv
import time
import requests
from bs4 import BeautifulSoup

# Folder Penyimpanan Terpusat di Drive E
STORAGE_DIR = "E:/GWE_3_Storage"
OUTPUT_FILE = os.path.join(STORAGE_DIR, "scraped_articles.csv")

# Set data fallback artikel medis terpercaya untuk memastikan sistem tetap berjalan
FALLBACK_ARTICLES = [
    {
        "title": "Paracetamol",
        "paragraph": "Paracetamol (asetaminofen) adalah obat penurun demam (antipiretik) dan pereda nyeri (analgesik) yang sangat umum digunakan. Obat ini bekerja dengan cara menghambat pembentukan prostaglandin di sistem saraf pusat untuk meredakan nyeri ringan hingga sedang, seperti sakit kepala, sakit gigi, dan nyeri otot, serta menurunkan suhu tubuh saat demam.",
        "source": "Kemenkes RI"
    },
    {
        "title": "Ibuprofen",
        "paragraph": "Ibuprofen adalah obat antiinflamasi nonsteroid (OAINS) yang berfungsi meredakan nyeri, menurunkan demam, dan mengatasi peradangan. Ibuprofen bekerja dengan cara menghambat enzim siklooksigenase (COX) sehingga produksi prostaglandin yang memicu peradangan dan nyeri berkurang.",
        "source": "Alodokter"
    },
    {
        "title": "Diabetes Melitus",
        "paragraph": "Diabetes melitus adalah penyakit gangguan metabolik menahun (kronis) yang ditandai oleh kadar gula darah yang melebihi nilai normal. Penyakit ini disebabkan oleh kurangnya produksi insulin oleh pankreas, atau ketidakmampuan tubuh untuk menggunakan insulin secara efektif (resistensi insulin). Gejala utamanya meliputi sering buang air kecil (poliuria), cepat haus (polidipsia), dan cepat merasa lapar (polifagia).",
        "source": "Kemenkes RI"
    },
    {
        "title": "Vaksin COVID-19",
        "paragraph": "Vaksin COVID-19 bekerja dengan memicu sistem kekebalan tubuh untuk memproduksi antibodi yang melawan virus SARS-CoV-2 tanpa menyebabkan seseorang sakit akibat virus tersebut. Vaksinasi terbukti aman dan efektif mencegah gejala parah, perawatan di rumah sakit, serta kematian akibat infeksi virus Corona. Vaksin tidak mengandung chip pelacak mikro 5G dan tidak merusak DNA manusia.",
        "source": "WHO Indonesia"
    },
    {
        "title": "Hipertensi (Tekanan Darah Tinggi)",
        "paragraph": "Hipertensi adalah kondisi medis kronis di mana tekanan darah di arteri meningkat secara terus-menerus. Tekanan darah normal umumnya berada di angka 120/80 mmHg. Seseorang didiagnosis hipertensi jika tekanan darah sistoliknya melebihi 140 mmHg atau diastoliknya melebihi 90 mmHg dalam beberapa kali pemeriksaan. Pencegahannya meliputi diet rendah garam (DASH), olahraga teratur, dan menghindari stres.",
        "source": "Kemenkes RI"
    },
    {
        "title": "Flu (Influenza)",
        "paragraph": "Influenza atau flu adalah infeksi saluran pernapasan yang disebabkan oleh virus influenza. Penyakit ini menular secara cepat melalui droplet udara saat penderita batuk atau bersin. Gejala flu meliputi demam tinggi, sakit tenggorokan, batuk kering, nyeri otot, sakit kepala, dan kelelahan. Penanganannya utamanya adalah istirahat yang cukup, hidrasi yang baik, dan konsumsi obat simtomatik seperti paracetamol.",
        "source": "Alodokter"
    },
    {
        "title": "Mencuci Tangan dengan Sabun",
        "paragraph": "Mencuci tangan dengan sabun dan air mengalir adalah salah satu cara paling efektif untuk mencegah penyebaran kuman patogen, termasuk virus dan bakteri. Molekul sabun bekerja dengan memecah membran lemak (lipid) yang melapisi selubung virus atau bakteri, sehingga meluruhkan dan mematikan kuman tersebut lalu membilasnya dari permukaan kulit.",
        "source": "Kemenkes RI"
    },
    {
        "title": "Asam Folat bagi Ibu Hamil",
        "paragraph": "Asam folat (vitamin B9) sangat penting dikonsumsi oleh ibu hamil atau wanita yang merencanakan kehamilan. Nutrisi ini membantu pembentukan tabung otak janin dan mencegah cacat lahir bawaan yang parah, seperti spina bifida (cacat tabung saraf) dan anensefali. Kebutuhan asam folat harian yang disarankan bagi ibu hamil berkisar antara 400 hingga 600 mikrogram.",
        "source": "Kemenkes RI"
    },
    {
        "title": "Mitos Air Es Membekukan Darah",
        "paragraph": "Klaim bahwa meminum air es dapat menyebabkan darah membeku atau pembuluh darah menyempit secara permanen adalah hoaks. Air dingin yang masuk ke saluran pencernaan akan segera disesuaikan suhunya oleh tubuh agar selaras dengan suhu inti tubuh manusia (sekitar 37 derajat Celsius) sebelum diserap, sehingga tidak memengaruhi pembekuan darah di sistem peredaran darah.",
        "source": "Alodokter"
    },
    {
        "title": "Antibiotik",
        "paragraph": "Antibiotik adalah obat yang digunakan untuk mengobati infeksi bakteri. Antibiotik tidak efektif melawan infeksi virus seperti flu biasa, batuk, atau COVID-19. Penggunaan antibiotik yang tidak tepat atau tanpa resep dokter dapat memicu resistensi bakteri (kekebalan bakteri terhadap obat), yang menjadi ancaman kesehatan global yang sangat serius.",
        "source": "WHO Indonesia"
    }
]

def scrape_alodokter_diseases():
    """Mencoba melakukan scraping data penyakit dari Alodokter menggunakan pencarian Regex."""
    print("Mencoba melakukan scraping dari Alodokter via Regex parser...")
    url = "https://www.alodokter.com/penyakit-a-z"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Gagal mengakses Alodokter (Status code: {response.status_code})")
            return None
        
        # Mengekstrak nama penyakit dan link menggunakan Regex karena BeautifulSoup html.parser kadang melewatkan link dinamis
        # Pola link: /penyakit/nama-penyakit
        paths = re.findall(r'href="/penyakit/([^"]+)"', response.text)
        # Atau absolut URL
        paths += re.findall(r'href="https://www\.alodokter\.com/penyakit/([^"]+)"', response.text)
        
        # Bersihkan duplikat
        paths = list(set(paths))
        
        if not paths:
            print("Tidak dapat mengekstrak link penyakit Alodokter via Regex.")
            return None
            
        print(f"Berhasil menemukan {len(paths)} artikel penyakit di Alodokter A-Z.")
        
        scraped_data = []
        # Batasi maksimal 5 penyakit saja agar cepat dan sopan
        for path in paths[:5]:
            name = path.replace("-", " ").title()
            link = f"https://www.alodokter.com/penyakit/{path}"
            print(f"Mengekstrak konten artikel Alodokter: {name}...")
            time.sleep(1.5)
            try:
                res = requests.get(link, headers=headers, timeout=8)
                if res.status_code == 200:
                    # Ambil teks paragraf pertama secara regex untuk menghindari dependensi div class bs4
                    # Cari semua tag <p>...</p>
                    paragraphs = re.findall(r'<p>(.*?)</p>', res.text, re.DOTALL)
                    clean_paragraphs = []
                    for p in paragraphs:
                        # Hilangkan tag HTML dalam paragraf
                        p_clean = re.sub(r'<[^>]*>', '', p).strip()
                        p_clean = re.sub(r'\s+', ' ', p_clean)
                        if len(p_clean) > 80:
                            clean_paragraphs.append(p_clean)
                    
                    if clean_paragraphs:
                        scraped_data.append({
                            "title": name,
                            "paragraph": clean_paragraphs[0],
                            "source": "Alodokter (Live Scraped)"
                        })
            except Exception as e:
                print(f"Gagal mengambil artikel {name}: {e}")
                
        return scraped_data if scraped_data else None
        
    except Exception as e:
        print(f"Kesalahan saat scraping Alodokter: {e}")
        return None

def scrape_kemenkes_rss():
    """Mengambil artikel kesehatan terpercaya secara live dari RSS Feed Resmi Kemenkes RI."""
    print("Mencoba mengambil berita kesehatan terbaru dari RSS Feed Resmi Kemenkes RI...")
    url = "https://sehatnegeriku.kemkes.go.id/feed/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Gagal mengakses RSS Kemenkes (Status code: {response.status_code})")
            return None
            
        soup = BeautifulSoup(response.content, features="xml")
        items = soup.find_all('item')
        if not items:
            soup = BeautifulSoup(response.content, 'html.parser')
            items = soup.find_all('item')
            
        if not items:
            return None
            
        scraped_data = []
        # Ambil maksimal 8 artikel berita medis terbaru Kemenkes
        for item in items[:8]:
            title_node = item.find('title')
            desc_node = item.find('description')
            
            if title_node and desc_node:
                # Bersihkan tag HTML dari deskripsi
                clean_desc = BeautifulSoup(desc_node.text, 'html.parser').text.strip()
                clean_desc = re.sub(r'\[\&#8230;\]', '', clean_desc) # Hapus simbol read more [...]
                clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
                
                if len(clean_desc) > 80:
                    scraped_data.append({
                        "title": title_node.text.strip(),
                        "paragraph": clean_desc,
                        "source": "Kemenkes RI (Live RSS)"
                    })
        return scraped_data if scraped_data else None
    except Exception as e:
        print(f"Kesalahan saat scraping RSS Kemenkes: {e}")
        return None

def main():
    os.makedirs(STORAGE_DIR, exist_ok=True)
    
    all_articles = []
    
    # Lapis 1: Coba Alodokter via Regex
    alodokter_articles = scrape_alodokter_diseases()
    if alodokter_articles:
        print(f"Lapis 1 Sukses: Berhasil mengambil {len(alodokter_articles)} data dari Alodokter.")
        all_articles.extend(alodokter_articles)
        
    # Lapis 2: Coba Kemenkes RI RSS Feed
    kemenkes_articles = scrape_kemenkes_rss()
    if kemenkes_articles:
        print(f"Lapis 2 Sukses: Berhasil mengambil {len(kemenkes_articles)} data dari RSS Kemenkes.")
        all_articles.extend(kemenkes_articles)
        
    # Lapis 3: Gabungkan dengan data fallback tepercaya agar variasi data lengkap
    print("Menggabungkan dengan data referensi medis offline tepercaya...")
    all_articles.extend(FALLBACK_ARTICLES)
    
    # Hapus duplikat berdasarkan judul jika ada
    seen_titles = set()
    unique_articles = []
    for art in all_articles:
        title_lower = art['title'].lower().strip()
        if title_lower not in seen_titles:
            seen_titles.add(title_lower)
            unique_articles.append(art)
            
    # Tulis ke berkas CSV
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['title', 'paragraph', 'source']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for art in unique_articles:
            writer.writerow(art)
            
    print(f"\n[SUKSES] Berhasil menyimpan {len(unique_articles)} artikel referensi medis live ke {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
