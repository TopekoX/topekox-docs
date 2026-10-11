# Dataset Penelitian untuk NLP
## Membangun Dataset Klasifikasi Tweet Permintaan Bantuan Darurat saat Bencana

> **Posisi dalam kurikulum:** setelah Fundamental NLP dan sebelum eksperimen klasifikasi TF-IDF/IndoBERT.  
> **Tujuan:** memahami cara merancang, mengumpulkan, memberi label, memvalidasi, dan membagi dataset penelitian NLP secara dapat dipertanggungjawabkan.

---

## 1. Tujuan Pembelajaran

Setelah mempelajari materi ini, Anda diharapkan mampu:

1. Menjelaskan perbedaan data, dataset, corpus, dan sampel.
2. Menentukan unit analisis dan tujuan klasifikasi.
3. Merancang skema label yang jelas.
4. Membuat struktur dataset yang konsisten.
5. Memahami proses pengumpulan data dan etika penggunaannya.
6. Melakukan anotasi dan pemeriksaan kualitas label.
7. Memeriksa duplikasi, data kosong, dan ketidakseimbangan kelas.
8. Membagi dataset menjadi train, validation, dan test tanpa kebocoran data.
9. Menyimpan dataset ke CSV dan membaca kembali menggunakan Pandas.

## 2. Posisi Dataset dalam Pipeline Penelitian NLP

Dataset bukan sekadar kumpulan teks. Dataset penelitian harus memiliki tujuan, definisi label, metadata yang relevan, dan prosedur pengumpulan yang terdokumentasi.

```text
Rumusan Masalah
      ↓
Definisi Unit Analisis
      ↓
Rancangan Skema Label
      ↓
Pengumpulan Data secara Sah
      ↓
Pembersihan Data dan Pemeriksaan Duplikasi
      ↓
Anotasi Label
      ↓
Quality Control
      ↓
Train / Validation / Test Split
      ↓
Baseline TF-IDF → Model Transformer/IndoBERT
      ↓
Evaluasi dan Analisis Kesalahan
```

Untuk contoh pembelajaran, kita akan memakai klasifikasi teks media sosial terkait bencana menjadi **Emergency** dan **Non-Emergency**. Label ini adalah contoh operasional untuk latihan; definisi final harus ditetapkan dan diuji sebelum digunakan pada penelitian nyata.

## 3. Istilah Penting

| Istilah | Arti | Contoh |
|---|---|---|
| Data | Fakta atau observasi yang dikumpulkan | Satu unggahan |
| Dataset | Kumpulan data dengan struktur tertentu | File CSV berisi unggahan dan label |
| Corpus | Kumpulan teks untuk analisis bahasa | Kumpulan teks unggahan berbahasa Indonesia |
| Sampel | Satu unit yang masuk ke analisis | Satu unggahan |
| Fitur | Informasi yang digunakan model | Kolom `text` |
| Label/target | Kategori yang diprediksi | Kolom `label` |
| Metadata | Informasi pendukung | Waktu, sumber, ID samaran |
| Anotasi | Proses menetapkan label | Penilai menentukan apakah teks merupakan permintaan bantuan darurat |

## 4. Rumuskan Tugas Klasifikasi dengan Tepat

Contoh pertanyaan penelitian:

> Seberapa baik model NLP mengidentifikasi unggahan berbahasa Indonesia yang berisi permintaan bantuan darurat terkait bencana?

### 4.1 Tentukan unit analisis

Unit analisis contoh ini adalah **satu unggahan teks**. Jika satu unggahan berisi beberapa kalimat, beri label berdasarkan keseluruhan isi unggahan menurut pedoman anotasi.

### 4.2 Tentukan definisi label

Contoh skema dua kelas:

| Label | Definisi operasional | Contoh ilustratif |
|---|---|---|
| `emergency` | Teks menunjukkan kebutuhan bantuan langsung atau situasi berbahaya yang membutuhkan respons segera | “Tolong evakuasi, air sudah masuk ke rumah dan kami terjebak.” |
| `non_emergency` | Teks memberi informasi, opini, atau percakapan terkait bencana tanpa permintaan bantuan langsung yang memenuhi definisi di atas | “Pemerintah mengimbau warga waspada terhadap banjir.” |

**Penting:** menyebut kata “banjir” tidak otomatis berarti emergency. Sebaliknya, permintaan bantuan dapat disampaikan tanpa memakai kata “tolong”. Label harus ditentukan dari konteks dan pedoman, bukan hanya kata kunci.

Untuk penelitian nyata, pertimbangkan kategori `uncertain` selama anotasi. Data yang ambigu sebaiknya ditinjau ulang, bukan dipaksa masuk ke salah satu kelas tanpa aturan.

### 4.3 Buat pedoman anotasi

Pedoman minimal mencakup:

- Definisi setiap label.
- Kriteria inklusi dan eksklusi.
- Cara menangani berita, kutipan, retweet, sarkasme, dan laporan pihak ketiga.
- Cara menangani unggahan yang menyebut lokasi tetapi tidak meminta bantuan.
- Aturan untuk teks ambigu atau konteks yang tidak lengkap.
- Contoh positif, negatif, dan kasus batas.

Lakukan uji coba anotasi pada sebagian kecil data. Revisi pedoman jika penilai sering berbeda pendapat.

## 5. Merancang Skema Dataset

Contoh kolom yang dapat digunakan:

| Kolom | Tipe | Kegunaan |
|---|---|---|
| `post_id` | string | ID internal yang tidak mengungkap identitas |
| `text` | string | Teks unggahan untuk klasifikasi |
| `label` | string | Label target |
| `event_id` | string | ID kejadian bencana untuk evaluasi berbasis kejadian |
| `posted_at` | datetime/string | Waktu unggahan jika tersedia dan sah digunakan |
| `source` | string | Sumber data |
| `annotator_id` | string | Kode penilai, jika diperlukan untuk audit |
| `split` | string | `train`, `validation`, atau `test` setelah pembagian |

Tidak semua penelitian memerlukan seluruh kolom. Simpan hanya metadata yang diperlukan, legal, dan etis. Jangan menyertakan nama asli, handle akun, nomor telepon, atau lokasi rinci seseorang kecuali benar-benar diperlukan dan memiliki dasar yang sesuai. Untuk contoh publikasi, gunakan ID samaran dan minimalkan informasi pribadi.

### Contoh CSV

```csv
post_id,text,label,event_id,posted_at
P001,"Tolong evakuasi, kami terjebak banjir",emergency,E001,2026-01-10T08:10:00
P002,"Pemda mengimbau warga waspada banjir",non_emergency,E001,2026-01-10T08:15:00
P003,"Mohon kirim air bersih ke pos pengungsian",emergency,E002,2026-02-02T12:30:00
P004,"Hujan deras mengguyur wilayah sejak pagi",non_emergency,E002,2026-02-02T12:35:00
```

**Data di atas sepenuhnya sintetis untuk pembelajaran, bukan unggahan nyata.**

## 6. Pengumpulan Data dan Etika

Sebelum mengumpulkan data, dokumentasikan:

1. Sumber data dan alasan pemilihannya.
2. Rentang waktu pengumpulan.
3. Kata kunci atau aturan pencarian.
4. Kriteria inklusi dan eksklusi.
5. Ketentuan layanan, izin, dan batas akses platform.
6. Cara menyimpan, mengamankan, dan menghapus data.
7. Risiko identifikasi ulang dan langkah mitigasinya.

Gunakan API resmi atau sumber yang memberi izin sesuai ketentuannya. Jangan mengakali pembatasan akses atau mengumpulkan data privat tanpa izin. Unggahan publik tetap dapat memuat informasi sensitif; pertimbangkan privasi, minimisasi data, dan persetujuan/peninjauan etik sesuai aturan institusi.

### Contoh kriteria

**Inklusi**
- Teks berbahasa Indonesia atau campuran yang memang termasuk cakupan studi.
- Relevan dengan kejadian bencana yang diteliti.
- Memiliki teks yang cukup untuk dinilai.

**Eksklusi**
- Duplikasi identik yang tidak dibutuhkan.
- Spam otomatis atau iklan yang tidak relevan.
- Teks kosong atau rusak.
- Konten di luar cakupan penelitian.

Jangan menghapus semua teks pendek secara otomatis: pesan singkat seperti “Tolong evakuasi!” bisa sangat relevan.

## 7. Anotasi Label

Anotasi adalah proses menetapkan label berdasarkan pedoman. Untuk meningkatkan kualitas:

1. Susun pedoman tertulis.
2. Lakukan pilot annotation pada sebagian data.
3. Jika memungkinkan, minta dua penilai memberi label secara independen pada subset yang sama.
4. Hitung kesepakatan antarpemberi label, misalnya Cohen’s kappa untuk dua penilai pada label kategorikal.
5. Periksa contoh yang berbeda label.
6. Selesaikan perbedaan melalui diskusi atau adjudikasi yang terdokumentasi.
7. Catat versi pedoman dan perubahan label.

Jangan menganggap label otomatis dari kata kunci sebagai ground truth. Kata kunci dapat membantu penyaringan awal, tetapi label penelitian perlu mengikuti pedoman yang konsisten.

## 8. Praktik Python: Membuat Dataset Contoh

Contoh berikut membuat dataset sintetis, menyimpannya ke CSV, lalu menampilkan hasilnya. Jangan menggunakan contoh kecil ini untuk menyimpulkan performa model penelitian.

### Kode

```python
import pandas as pd

data = [
    {
        "post_id": "P001",
        "text": "Tolong evakuasi, kami terjebak banjir",
        "label": "emergency",
        "event_id": "E001",
    },
    {
        "post_id": "P002",
        "text": "Pemda mengimbau warga waspada banjir",
        "label": "non_emergency",
        "event_id": "E001",
    },
    {
        "post_id": "P003",
        "text": "Mohon kirim air bersih ke pos pengungsian",
        "label": "emergency",
        "event_id": "E002",
    },
    {
        "post_id": "P004",
        "text": "Hujan deras mengguyur wilayah sejak pagi",
        "label": "non_emergency",
        "event_id": "E002",
    },
    {
        "post_id": "P005",
        "text": "Kami butuh pertolongan, lansia belum dievakuasi",
        "label": "emergency",
        "event_id": "E003",
    },
    {
        "post_id": "P006",
        "text": "Berikut informasi prakiraan cuaca hari ini",
        "label": "non_emergency",
        "event_id": "E003",
    },
]

df = pd.DataFrame(data)

df.to_csv("dataset_bencana_sintetis.csv", index=False)

print("Ukuran dataset:", df.shape)
print("\nData:")
print(df.to_string(index=False))
```

### Output

```text
Ukuran dataset: (6, 4)

Data:
post_id                                              text          label event_id
   P001              Tolong evakuasi, kami terjebak banjir      emergency     E001
   P002              Pemda mengimbau warga waspada banjir  non_emergency     E001
   P003       Mohon kirim air bersih ke pos pengungsian      emergency     E002
   P004           Hujan deras mengguyur wilayah sejak pagi  non_emergency     E002
   P005 Kami butuh pertolongan, lansia belum dievakuasi      emergency     E003
   P006        Berikut informasi prakiraan cuaca hari ini  non_emergency     E003
```

File `dataset_bencana_sintetis.csv` akan dibuat pada direktori kerja saat kode dijalankan.

## 9. Memeriksa Kualitas Dataset

Sebelum modeling, lakukan pemeriksaan awal.

### Kode

```python
import pandas as pd

df = pd.read_csv("dataset_bencana_sintetis.csv")

print("Jumlah baris:", len(df))
print("Jumlah nilai kosong per kolom:")
print(df.isna().sum())

print("\nJumlah duplikasi teks:", df["text"].duplicated().sum())

print("\nDistribusi label:")
print(df["label"].value_counts())

print("\nDistribusi label dalam proporsi:")
print(df["label"].value_counts(normalize=True).round(3))
```

### Output

```text
Jumlah baris: 6
Jumlah nilai kosong per kolom:
post_id     0
text        0
label       0
event_id    0
dtype: int64

Jumlah duplikasi teks: 0

Distribusi label:
label
emergency        3
non_emergency    3

Distribusi label dalam proporsi:
label
emergency        0.5
non_emergency    0.5
```

Interpretasi:
- Tidak ada nilai kosong pada contoh.
- Tidak ada teks duplikat yang identik.
- Kedua kelas berjumlah sama pada contoh ini.
- Dataset sintetis ini terlalu kecil untuk melatih atau menilai model secara bermakna.

Pada dataset nyata, periksa juga duplikasi setelah normalisasi, ID yang berulang, label yang tidak valid, distribusi berdasarkan kejadian, serta kualitas teks. Jangan menghapus duplikasi lintas kejadian secara membabi buta: teks yang sama bisa memiliki konteks berbeda, tetapi salinan yang bocor ke split berbeda juga dapat membuat evaluasi terlalu optimistis.

## 10. Pemeriksaan Label dan Teks

Pastikan nilai label sesuai dengan skema yang ditentukan.

```python
allowed_labels = {"emergency", "non_emergency"}

invalid_labels = set(df["label"].dropna().unique()) - allowed_labels
print("Label tidak valid:", invalid_labels)

empty_text = df["text"].fillna("").str.strip().eq("")
print("Teks kosong:", int(empty_text.sum()))
```

### Output

```text
Label tidak valid: set()
Teks kosong: 0
```

Catatan: gunakan pemeriksaan ini sebagai validasi awal. Pemeriksaan otomatis tidak dapat menentukan apakah label benar secara semantik; itu tetap memerlukan pedoman dan peninjauan anotasi.

## 11. Membagi Dataset: Train, Validation, dan Test

Ketiga bagian mempunyai fungsi berbeda:

| Split | Fungsi |
|---|---|
| Train | Melatih parameter model |
| Validation | Memilih konfigurasi dan hyperparameter |
| Test | Evaluasi akhir pada data yang tidak digunakan saat pengembangan |

Pembagian acak biasa dapat menyebabkan kebocoran jika unggahan dari kejadian bencana yang sama muncul pada train dan test. Untuk tujuan generalisasi ke kejadian baru, pertimbangkan **event-based split**: semua unggahan dari satu `event_id` ditempatkan hanya pada satu split.

Contoh pembagian sederhana berdasarkan kejadian menggunakan scikit-learn:

```python
from sklearn.model_selection import GroupShuffleSplit

# Dataset ini hanya contoh sintetik dengan 3 kejadian.
# Pada riset nyata, gunakan jauh lebih banyak kejadian.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=1/3,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(df, y=df["label"], groups=df["event_id"])
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

print("Train events:", sorted(train_df["event_id"].unique()))
print("Test events:", sorted(test_df["event_id"].unique()))
print("Overlap event:", set(train_df["event_id"]) & set(test_df["event_id"]))
print("Jumlah train:", len(train_df))
print("Jumlah test:", len(test_df))
```

### Output yang diharapkan

Output persis bergantung pada versi pustaka dan pembagian grup yang dipilih. Syarat pentingnya:

```text
Overlap event: set()
```

Dengan hanya tiga kejadian, pembagian ini tidak cukup untuk evaluasi yang stabil. Untuk penelitian nyata, kumpulkan jumlah kejadian yang memadai dan tentukan strategi split sebelum membandingkan model. Jika perlu train/validation/test, buat ketiganya berdasarkan grup kejadian, bukan dengan membagi ulang baris secara acak.

## 12. Data Leakage

**Data leakage** terjadi ketika informasi dari data evaluasi masuk ke proses pelatihan atau pemilihan model, sehingga skor terlihat lebih baik daripada kemampuan generalisasi sebenarnya.

Contoh yang perlu dihindari:
- Teks yang sama atau hampir sama berada di train dan test.
- Unggahan dari satu kejadian yang sama tersebar pada semua split, padahal tujuan penelitian adalah generalisasi ke kejadian baru.
- TF-IDF di-`fit` pada seluruh dataset sebelum split.
- Memilih hyperparameter berdasarkan hasil test berulang kali.
- Menggunakan informasi yang baru tersedia setelah waktu prediksi untuk tugas deteksi dini.

Praktik yang benar untuk baseline:

```python
# Split data terlebih dahulu.
# Fit vectorizer hanya pada data train.
X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])
```

Untuk IndoBERT, tokenizer pretrained biasanya sudah memiliki vocabulary sendiri. Namun, prinsip pemisahan data tetap berlaku: jangan memakai label test untuk melatih atau memilih model.

## 13. Ukuran Dataset dan Ketidakseimbangan Kelas

Tidak ada satu angka universal yang menjamin dataset cukup. Kebutuhan ukuran bergantung pada keragaman bahasa, jumlah kelas, kejadian, kualitas anotasi, dan tujuan evaluasi.

Yang perlu dilaporkan:
- Jumlah total sampel.
- Jumlah dan proporsi per kelas.
- Jumlah kejadian bencana yang berbeda.
- Rentang waktu dan wilayah cakupan.
- Jumlah sampel yang dikeluarkan dan alasannya.
- Prosedur anotasi dan kesepakatan penilai.
- Distribusi label pada setiap split.

Jika kelas `emergency` jauh lebih sedikit, accuracy saja dapat menyesatkan. Laporkan precision, recall, F1 per kelas, macro F1, confusion matrix, dan metrik yang relevan dengan tujuan sistem.

## 14. Checklist Sebelum Modeling

- [ ] Tujuan klasifikasi ditulis dengan jelas.
- [ ] Unit analisis ditetapkan.
- [ ] Definisi label dan pedoman anotasi tersedia.
- [ ] Sumber dan metode pengumpulan terdokumentasi.
- [ ] Ketentuan penggunaan data dan privasi diperiksa.
- [ ] Nilai kosong, duplikasi, dan label tidak valid diperiksa.
- [ ] Kualitas anotasi diperiksa.
- [ ] Distribusi label dan kejadian diringkas.
- [ ] Train/validation/test split ditentukan tanpa kebocoran.
- [ ] Test set tidak digunakan untuk memilih model.
- [ ] Dataset dan versi pedoman anotasi dicatat.

## 15. Latihan Mandiri

1. Tambahkan minimal 10 contoh teks sintetis dengan label yang sesuai.
2. Buat dua contoh ambigu dan tuliskan alasan mengapa perlu peninjauan.
3. Hitung distribusi label dalam jumlah dan persentase.
4. Buat pemeriksaan untuk `post_id` yang duplikat.
5. Jelaskan mengapa pembagian berdasarkan `event_id` dapat lebih tepat daripada pembagian acak baris untuk menguji generalisasi ke kejadian baru.
6. Tulis satu paragraf tentang keterbatasan dataset sintetis dibandingkan data nyata.

### Tantangan lanjutan

Rancang skema dataset untuk tiga kelas:

- `emergency`: kebutuhan bantuan langsung atau bahaya yang membutuhkan respons segera.
- `informational`: informasi atau laporan bencana tanpa permintaan bantuan langsung.
- `unrelated`: tidak relevan dengan bencana yang diteliti.

Tuliskan pedoman untuk membedakan `emergency` dan `informational`, termasuk contoh kasus batas.

## 16. Ringkasan

Dataset penelitian yang baik bukan hanya besar. Dataset harus sesuai dengan pertanyaan penelitian, memiliki label yang konsisten, dikumpulkan secara etis, didokumentasikan, dan dievaluasi tanpa kebocoran data.

Untuk proyek klasifikasi permintaan bantuan darurat:
1. Definisikan label berdasarkan kebutuhan bantuan, bukan sekadar kata kunci bencana.
2. Dokumentasikan pengumpulan dan anotasi.
3. Periksa kualitas data dan ketidakseimbangan kelas.
4. Pertimbangkan pembagian berdasarkan kejadian untuk menguji generalisasi.
5. Bangun baseline TF-IDF terlebih dahulu, lalu bandingkan dengan IndoBERT.

**Checkpoint:** Anda siap melanjutkan ke eksperimen klasifikasi ketika mampu menjelaskan asal data, arti setiap label, kualitas anotasi, dan alasan pemilihan strategi pembagian dataset.
