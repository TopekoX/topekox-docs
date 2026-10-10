# Deteksi Tweet Permintaan Bantuan Darurat pada Saat Bencana Menggunakan IndoBERT

Sistem klasifikasi teks berbahasa Indonesia untuk membantu mengidentifikasi tweet yang mengindikasikan permintaan bantuan darurat saat bencana. Penelitian ini membandingkan pendekatan *classical machine learning* berbasis TF-IDF dengan model Transformer yang di-*fine-tune*, yaitu IndoBERT. Model terpilih direncanakan tersedia melalui REST API menggunakan FastAPI dan dapat diintegrasikan ke website informasi bencana.

> **Status proyek:** Research prototype / dalam pengembangan. Hasil eksperimen, performa model, dan ketersediaan data akan diisi setelah penelitian dilakukan. Sistem ini bukan pengganti layanan resmi tanggap darurat.

## Daftar Isi

- [Latar Belakang](#latar-belakang)
- [Tujuan](#tujuan)
- [Ruang Lingkup](#ruang-lingkup)
- [Metodologi](#metodologi)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Dataset](#dataset)
- [Eksperimen dan Evaluasi](#eksperimen-dan-evaluasi)
- [Teknologi](#teknologi)
- [Struktur Direktori](#struktur-direktori)
- [Instalasi](#instalasi)
- [Menjalankan API](#menjalankan-api)
- [Contoh Penggunaan API](#contoh-penggunaan-api)
- [Rencana Pengembangan](#rencana-pengembangan)
- [Etika dan Batasan](#etika-dan-batasan)
- [Sitasi](#sitasi)
- [Lisensi](#lisensi)

## Latar Belakang

Media sosial dapat memuat informasi tentang kondisi bencana dan permintaan bantuan dari masyarakat. Namun, unggahan yang relevan bercampur dengan berita, opini, percakapan umum, dan informasi yang tidak menunjukkan kebutuhan bantuan secara langsung. Klasifikasi teks dapat membantu menyaring unggahan yang berpotensi membutuhkan perhatian lebih lanjut.

Penelitian ini mengkaji penggunaan IndoBERT untuk mengklasifikasikan tweet berbahasa Indonesia ke dalam dua kelas awal:

- **Emergency** — teks yang mengindikasikan permintaan bantuan atau kebutuhan darurat terkait bencana.
- **Non-Emergency** — teks yang tidak memenuhi kriteria permintaan bantuan darurat berdasarkan pedoman anotasi penelitian.

Keluaran model merupakan hasil prediksi otomatis dan harus ditinjau manusia sebelum digunakan untuk pengambilan keputusan operasional.

## Tujuan

1. Menyusun dataset tweet bencana berbahasa Indonesia dengan pedoman anotasi yang jelas.
2. Membangun baseline klasifikasi menggunakan TF-IDF + Logistic Regression dan TF-IDF + SVM.
3. Melakukan *fine-tuning* IndoBERT untuk klasifikasi `Emergency` dan `Non-Emergency`.
4. Membandingkan model menggunakan metrik yang sesuai, terutama Recall kelas `Emergency` dan F1-score.
5. Menyediakan model terpilih melalui REST API berbasis FastAPI sebagai prototipe integrasi website informasi bencana.

## Ruang Lingkup

- **Bahasa:** Bahasa Indonesia, termasuk variasi bahasa informal media sosial.
- **Sumber data:** unggahan Twitter/X terkait bencana yang dikumpulkan secara sah sesuai ketentuan platform dan kebijakan yang berlaku.
- **Tugas utama:** klasifikasi biner `Emergency` vs. `Non-Emergency`.
- **Model pembanding:** TF-IDF + Logistic Regression, TF-IDF + SVM, dan IndoBERT yang di-*fine-tune*.
- **Produk:** REST API dan prototipe dashboard/website informasi bencana.
- **Di luar cakupan awal:** pengiriman petugas otomatis, verifikasi lokasi secara pasti, serta klaim bahwa prediksi model merupakan laporan darurat yang telah terverifikasi.

Klasifikasi jenis bantuan seperti makanan, air, medis, evakuasi, penyelamatan, dan tempat berlindung dapat dipertimbangkan sebagai pengembangan berikutnya, bukan bagian wajib dari klasifikasi biner awal.

## Metodologi

Alur penelitian yang direncanakan:

1. **Pengumpulan data** — mengumpulkan tweet terkait satu atau beberapa jenis kejadian bencana sesuai ruang lingkup yang ditetapkan.
2. **Penyaringan dan deduplikasi** — menghapus duplikasi dan menyaring data yang tidak relevan.
3. **Anotasi** — memberi label berdasarkan pedoman tertulis; kasus ambigu ditinjau ulang.
4. **Audit dataset** — memeriksa distribusi kelas, kualitas label, data kosong, dan potensi bias.
5. **Pembagian dataset** — memisahkan train, validation, dan test sebelum proses yang dapat mempelajari karakteristik data. Data duplikat atau sangat mirip harus dicegah tersebar antar-split.
6. **Baseline** — melatih TF-IDF + Logistic Regression dan TF-IDF + SVM.
7. **Fine-tuning IndoBERT** — melakukan tokenisasi dan fine-tuning untuk klasifikasi biner.
8. **Evaluasi** — membandingkan performa pada test set yang sama dan tidak digunakan untuk pemilihan model.
9. **Analisis kesalahan** — meninjau false negative, false positive, kasus ambigu, dan performa per kelas.
10. **Deployment prototipe** — menyediakan model melalui FastAPI dan menyimpan hasil sesuai kebutuhan aplikasi.

### Catatan preprocessing

Preprocessing harus disesuaikan dengan pendekatan model. Untuk TF-IDF, eksperimen dapat mencakup normalisasi teks, tokenisasi, dan pengelolaan kata umum. Untuk IndoBERT, hindari menghapus kata atau mengubah teks secara agresif tanpa eksperimen karena konteks kalimat dapat hilang. Gunakan tokenizer bawaan model IndoBERT yang dipilih.

## Arsitektur Sistem

```text
Tweet terkait bencana
        |
        v
Penyaringan, deduplikasi, dan anotasi
        |
        v
Dataset berlabel
        |
        +-------------------------------+
        |                               |
        v                               v
TF-IDF + Logistic Regression       TF-IDF + SVM
        |                               |
        +---------------+---------------+
                        |
                        v
               Evaluasi baseline
                        |
                        v
                Fine-tuning IndoBERT
                        |
                        v
              Evaluasi dan pemilihan model
                        |
                        v
                  FastAPI REST API
                        |
                        v
              Website / dashboard prototipe
```

Alur pengembangan model dan alur prediksi produksi merupakan dua proses yang berbeda. Model hanya dipublikasikan ke API setelah artefak, konfigurasi label, dan preprocessing/tokenizer yang sesuai disimpan bersama.

## Dataset

Dataset penelitian belum diasumsikan tersedia atau telah selesai dikumpulkan. Detail berikut harus dicatat setelah proses pengumpulan data.

| Informasi | Keterangan |
| --- | --- |
| Platform | Twitter/X |
| Bahasa | Bahasa Indonesia |
| Jenis bencana | Ditentukan berdasarkan ruang lingkup penelitian |
| Unit data | Teks unggahan/tweet |
| Label | `Emergency`, `Non-Emergency` |
| Jumlah data | Diisi setelah pengumpulan dan kurasi |
| Rentang waktu | Diisi sesuai protokol pengumpulan |
| Pedoman anotasi | Didokumentasikan dan digunakan secara konsisten |
| Lisensi/ketentuan penggunaan | Mengikuti ketentuan sumber dan platform |

Contoh ilustratif (bukan data penelitian aktual):

| Teks | Label ilustratif |
| --- | --- |
| “Tolong, keluarga kami terjebak banjir dan membutuhkan evakuasi.” | `Emergency` |
| “Banjir terjadi di beberapa wilayah sejak pagi.” | `Non-Emergency` |

Label harus ditentukan berdasarkan pedoman anotasi, bukan hanya keberadaan kata seperti “banjir”, “tolong”, atau “bantuan”. Tweet informatif tentang bencana tidak otomatis merupakan permintaan bantuan darurat.

## Eksperimen dan Evaluasi

Model yang direncanakan untuk dibandingkan:

1. TF-IDF + Logistic Regression
2. TF-IDF + SVM
3. IndoBERT *fine-tuning*

Metrik evaluasi:

- **Precision:** proporsi prediksi positif yang benar.
- **Recall:** proporsi data positif aktual yang berhasil ditemukan.
- **F1-score:** keseimbangan precision dan recall.
- **Macro F1:** rata-rata F1 antarkelas dengan bobot kelas setara.
- **Confusion matrix:** memperlihatkan TP, TN, FP, dan FN.

Recall kelas `Emergency` menjadi metrik penting karena false negative berarti unggahan yang memenuhi kriteria darurat tidak terdeteksi. Namun, precision juga perlu dipantau agar sistem tidak menghasilkan terlalu banyak alarm keliru.

| Model | Accuracy | Precision Emergency | Recall Emergency | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | Belum diukur | Belum diukur | Belum diukur | Belum diukur |
| TF-IDF + SVM | Belum diukur | Belum diukur | Belum diukur | Belum diukur |
| IndoBERT fine-tuning | Belum diukur | Belum diukur | Belum diukur | Belum diukur |

Tidak ada hasil performa yang diasumsikan pada README ini. Isi tabel setelah eksperimen selesai. Semua model harus dievaluasi pada test set yang sama, dan test set tidak boleh digunakan untuk tuning hyperparameter atau memilih checkpoint.

## Teknologi

Teknologi yang direncanakan:

- **Python** — bahasa pemrograman.
- **Pandas / NumPy** — pengolahan data.
- **scikit-learn** — TF-IDF, baseline, dan evaluasi.
- **PyTorch** — framework deep learning.
- **Hugging Face Transformers** — tokenizer dan model IndoBERT.
- **FastAPI** — REST API.
- **PostgreSQL** — penyimpanan data/prediksi jika dibutuhkan.
- **Frontend web** — dashboard sebagai lapisan presentasi; framework dapat ditetapkan saat implementasi.

Versi paket dan model IndoBERT yang benar-benar digunakan harus dicatat dalam file dependensi serta dokumentasi eksperimen.

## Struktur Direktori

Contoh struktur awal yang dapat disesuaikan:

```text
disaster-emergency-nlp/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/                 # Data mentah; jangan commit data sensitif
│   ├── processed/
│   └── README.md            # Sumber, skema, dan aturan akses data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_tfidf_baselines.ipynb
│   └── 04_indobert_finetuning.ipynb
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── evaluation/
│   └── api/
│       └── main.py
├── tests/
├── artifacts/               # Model lokal; biasanya tidak di-commit
└── reports/
    └── experiment_results/
```

## Instalasi

Disarankan menggunakan Python virtual environment. Perintah berikut merupakan contoh setup awal; sesuaikan versi paket dengan lingkungan eksperimen Anda.

```bash
python -m venv .venv
```

Aktifkan environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS
source .venv/bin/activate
```

Instal paket dasar:

```bash
pip install pandas numpy scikit-learn jupyter
pip install torch transformers datasets evaluate accelerate
pip install fastapi "uvicorn[standard]"
```

Setelah lingkungan stabil, simpan versi dependensi:

```bash
pip freeze > requirements.txt
```

Untuk training IndoBERT, GPU yang kompatibel dapat mempercepat proses, tetapi eksperimen kecil masih dapat dilakukan di CPU dengan waktu komputasi lebih lama. Ikuti petunjuk instalasi PyTorch yang sesuai dengan perangkat keras Anda.

## Menjalankan API

Setelah modul `src/api/main.py` dan pemuatan model selesai diimplementasikan, jalankan:

```bash
uvicorn src.api.main:app --reload
```

Dokumentasi interaktif FastAPI biasanya tersedia di:

```text
http://127.0.0.1:8000/docs
```

Perintah tersebut adalah contoh target implementasi; API belum dianggap berjalan sebelum modul dan model benar-benar tersedia.

## Contoh Penggunaan API

Rancangan endpoint prediksi: `POST /predict`.

Request ilustratif:

```json
{
  "text": "Tolong bantu kami, rumah sudah terendam banjir."
}
```

Response ilustratif:

```json
{
  "label": "Emergency",
  "confidence": 0.94
}
```

**Response di atas hanya contoh format, bukan hasil model aktual.** Nilai confidence harus berasal dari inferensi model yang berjalan. Confidence bukan jaminan bahwa laporan benar atau telah diverifikasi. API nyata juga perlu menangani input kosong, batas panjang teks, kesalahan model, dan validasi skema.

## Rencana Pengembangan

- [ ] Menetapkan ruang lingkup bencana dan protokol pengumpulan data.
- [ ] Menyusun pedoman anotasi dan contoh kasus ambigu.
- [ ] Mengumpulkan, membersihkan, dan mendokumentasikan dataset.
- [ ] Memeriksa distribusi kelas dan kualitas anotasi.
- [ ] Menetapkan train/validation/test split tanpa kebocoran data.
- [ ] Membangun baseline TF-IDF + Logistic Regression.
- [ ] Membangun baseline TF-IDF + SVM.
- [ ] Melakukan fine-tuning IndoBERT.
- [ ] Mengevaluasi model dengan metrik per kelas.
- [ ] Melakukan analisis kesalahan dan uji robustness.
- [ ] Menyimpan model, tokenizer, konfigurasi, dan hasil eksperimen.
- [ ] Membuat REST API FastAPI.
- [ ] Mengintegrasikan API ke prototipe dashboard.
- [ ] Menulis laporan penelitian dan mendokumentasikan keterbatasan.

## Etika dan Batasan

- Gunakan data sesuai ketentuan layanan, peraturan, dan izin yang berlaku.
- Hindari memublikasikan informasi pribadi, identitas akun, nomor telepon, atau lokasi rinci yang dapat membahayakan pengguna.
- Simpan data mentah secara aman dan batasi akses.
- Dokumentasikan sumber data, metode pengambilan, proses anonimisasi, dan batasan penggunaan.
- Prediksi model dapat salah, bias, atau gagal memahami konteks, sarkasme, dan bahasa lokal.
- Sistem ini merupakan alat bantu penyaringan informasi, bukan pengganti verifikasi manusia atau kanal resmi layanan darurat.
- Jangan menyatakan lokasi, urgensi, atau kebutuhan bantuan sebagai fakta terverifikasi hanya berdasarkan prediksi model.

## Sitasi

Cantumkan sitasi untuk model IndoBERT, library, dataset, dan metode yang benar-benar digunakan. Sebagai referensi awal, periksa publikasi dan dokumentasi resmi IndoBERT serta Hugging Face Transformers. Tambahkan sitasi bibliografis final setelah model dan versi yang dipakai dipastikan.

## Lisensi

Lisensi proyek belum ditentukan. Sebelum memilih lisensi, pastikan hak penggunaan kode, dataset, dan bobot model pihak ketiga telah diperiksa. Lisensi kode proyek tidak otomatis memberikan hak atas data atau model eksternal.
