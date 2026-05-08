# 🎓 Sistem Rekomendasi Karier untuk Siswa

Proyek Akhir Machine Learning Terapan – Dicoding.

Aplikasi web yang merekomendasikan karier kepada siswa SMA berdasarkan profil akademik mereka (nilai mata pelajaran, jam belajar, ekstrakurikuler, dll) menggunakan **Content-Based Filtering** dengan algoritma cosine similarity.

## 🚀 Demo

🔗 **Link Aplikasi**: [https://your-app-name.streamlit.app](#) *(akan diisi setelah deploy)*

## 📊 Dataset

Dataset berasal dari Kaggle:  
[Student Studeis Recommendation](https://www.kaggle.com/datasets/noorsaeed/student-studeis-recommendation)

- 2.000 siswa
- 17 kolom (nilai 7 mapel + profil + cita-cita karier)
- 17 jenis cita-cita karier

## 🤖 Model

**Content-Based Filtering** dengan:
- Cosine similarity antar profil siswa
- Weighted voting dari top-50 nearest neighbors
- 12 fitur: gender, part-time job, absen, ekskul, jam belajar, 7 nilai mata pelajaran

**Performa**:
- Hit Rate Top-5: 76.76%
- Hit Rate Top-7: 86.44%
- Hit Rate Top-10: 94.60%

## 🛠️ Cara Menjalankan Lokal

```bash
# 1. Clone repo
git clone https://github.com/username/career-recommender-streamlit.git
cd career-recommender-streamlit

# 2. Buat virtual environment (opsional tapi disarankan)
python -m venv venv
source venv/bin/activate     # Linux/Mac
# venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run aplikasi
streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501`

## 📁 Struktur File

```
career-recommender-streamlit/
├── app.py                  # Aplikasi Streamlit utama
├── requirements.txt        # Dependencies Python
├── student-scores.csv      # Dataset
├── README.md               # Dokumentasi
└── .gitignore              # File yg di-ignore Git
```

## 🎯 Fitur Aplikasi

1. **🏠 Beranda** — overview aplikasi dan statistik dataset
2. **🔍 Cari Siswa di Dataset** — pilih siswa yang ada, lihat rekomendasi karier
3. **✏️ Input Profil Siswa Baru** — masukkan profil siswa baru, dapatkan rekomendasi instant
4. **📊 Eksplorasi Dataset** — lihat statistik dan distribusi data

## 📜 Lisensi

MIT License – Bebas dipakai untuk keperluan edukasi.

## 👤 Author

Proyek Akhir Dicoding Machine Learning Terapan
