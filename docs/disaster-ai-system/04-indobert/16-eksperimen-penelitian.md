# Eksperimen Penelitian NLP: Klasifikasi Tweet Permintaan Bantuan Darurat

## 1. Tujuan Pembelajaran

Materi ini membahas cara merancang dan menjalankan eksperimen klasifikasi teks untuk penelitian **klasifikasi tweet permintaan bantuan darurat saat bencana**. Fokus tahap ini adalah membangun dan membandingkan model baseline berbasis TF-IDF dan machine learning klasik.

Setelah menyelesaikan materi, Anda diharapkan mampu:

1. Merumuskan tujuan dan pertanyaan eksperimen.
2. Menyiapkan dataset berlabel secara benar.
3. Membuat pembagian data train, validation, dan test tanpa kebocoran data.
4. Membandingkan TF-IDF + Logistic Regression dan TF-IDF + Linear SVM.
5. Mengukur Accuracy, Precision, Recall, F1-score, dan confusion matrix.
6. Menyimpan hasil eksperimen agar dapat direproduksi.
7. Menyiapkan baseline yang kelak dapat dibandingkan dengan IndoBERT pada penelitian tesis.

> **Batasan lingkup:** materi ini menggunakan TF-IDF dan model machine learning klasik untuk baseline penelitian. Fine-tuning IndoBERT merupakan eksperimen lanjutan yang dapat dilakukan pada tahap tesis. Hasil contoh pada materi ini bersifat ilustratif; hasil nyata harus berasal dari dataset dan eksekusi Anda sendiri.

## 2. Posisi Eksperimen dalam Penelitian

Eksperimen bukan sekadar menjalankan beberapa algoritma. Eksperimen harus dirancang agar perbandingan adil, dapat diulang, dan menjawab pertanyaan penelitian.

Alur umum:

```text
Rumusan Masalah
      ↓
Dataset dan Pedoman Label
      ↓
Pemeriksaan Kualitas Data
      ↓
Train / Validation / Test Split
      ↓
TF-IDF + Logistic Regression
      ↓
TF-IDF + Linear SVM
      ↓
Evaluasi dengan Metrik yang Sama
      ↓
Analisis Kesalahan
      ↓
Kesimpulan dan Pelaporan
```

## 3. Rumusan Masalah dan Pertanyaan Eksperimen

Contoh rumusan masalah:

> Seberapa baik metode TF-IDF yang dikombinasikan dengan machine learning klasik dalam mengklasifikasikan tweet berbahasa Indonesia yang mengandung permintaan bantuan darurat saat bencana?

Contoh pertanyaan eksperimen:

- **RQ1:** Bagaimana kinerja TF-IDF + Logistic Regression dalam mendeteksi tweet permintaan bantuan darurat?
- **RQ2:** Bagaimana kinerja TF-IDF + Linear SVM pada dataset dan pembagian data yang sama?
- **RQ3:** Model mana yang memberikan keseimbangan precision dan recall terbaik untuk kelas `Emergency`?
- **RQ4:** Jenis kesalahan apa yang paling sering terjadi, terutama false negative pada kelas `Emergency`?

Hipotesis awal boleh dibuat, tetapi jangan menganggap salah satu model pasti menang sebelum eksperimen dilakukan.

## 4. Definisi Label dan Unit Analisis

Untuk eksperimen awal, gunakan klasifikasi biner:

| Label | Makna | Contoh ilustratif |
|---|---|---|
| `Emergency` | Tweet menunjukkan kebutuhan bantuan langsung atau kondisi darurat yang memerlukan respons | “Tolong evakuasi, air sudah masuk ke rumah.” |
| `Non-Emergency` | Tweet membahas bencana atau topik terkait, tetapi tidak menunjukkan permintaan bantuan langsung menurut pedoman anotasi | “Pemerintah mengumumkan status siaga banjir.” |

Contoh tersebut hanya ilustrasi. Penentuan label harus mengikuti **pedoman anotasi tertulis**. Tweet yang menyebut banjir tidak otomatis menjadi `Emergency`; konteks dan kebutuhan bantuan harus diperiksa.

Tentukan sejak awal:
- Apakah satu baris mewakili satu tweet?
- Apakah retweet atau duplikat dihapus?
- Bagaimana menangani tweet ambigu, sarkasme, berita, dan informasi lokasi?
- Apakah tweet tanpa konteks yang cukup diberi label khusus untuk ditinjau atau dikeluarkan dari eksperimen?

Simpan label sebagai nilai konsisten, misalnya `Emergency` dan `Non-Emergency`.

## 5. Struktur Dataset

Siapkan file CSV dengan sekurangnya dua kolom:

| Kolom | Isi |
|---|---|
| `text` | Teks tweet |
| `label` | Label kelas |

Contoh `data/tweets.csv`:

```csv
text,label
"Tolong bantu kami, rumah terendam banjir",Emergency
"Perlu evakuasi warga di sekitar sungai",Emergency
"BPBD mengimbau warga tetap waspada terhadap banjir",Non-Emergency
"Hujan turun sejak pagi di wilayah kota",Non-Emergency
```

Dataset di atas hanya contoh kecil untuk menjelaskan format. **Empat baris tersebut tidak cukup untuk menghasilkan evaluasi penelitian yang bermakna.**

Struktur proyek yang disarankan:

```text
eksperimen-nlp/
├── data/
│   └── tweets.csv
├── outputs/
│   ├── classification_report_logreg.txt
│   ├── classification_report_svm.txt
│   └── model_comparison.csv
└── experiment.py
```

Jangan memasukkan informasi pribadi yang tidak diperlukan. Pastikan pengumpulan dan penggunaan data mengikuti ketentuan platform, etika penelitian, dan aturan privasi yang berlaku.

## 6. Pemeriksaan Data Sebelum Eksperimen

Sebelum melatih model, periksa:

1. Nilai kosong pada teks dan label.
2. Label yang tidak dikenal.
3. Duplikat teks.
4. Distribusi jumlah setiap kelas.
5. Contoh teks dari masing-masing kelas.
6. Potensi duplikasi atau kemiripan tinggi lintas train dan test.

Duplikat dapat membuat hasil evaluasi tampak lebih baik dari kemampuan generalisasi yang sebenarnya. Jika beberapa tweet berasal dari percakapan, peristiwa, atau sumber yang sama, pertimbangkan pemisahan berdasarkan grup atau waktu agar informasi yang hampir sama tidak bocor ke test set.

## 7. Desain Eksperimen yang Adil

Gunakan dataset dan pembagian data yang sama untuk setiap model.

### 7.1 Pembagian data

Salah satu rancangan awal adalah:

- 70% train
- 15% validation
- 15% test

`Train` digunakan untuk melatih model. `Validation` digunakan untuk memilih konfigurasi. `Test` digunakan sekali untuk evaluasi akhir setelah keputusan model selesai dibuat.

Untuk dataset yang kecil, pembagian ini mungkin kurang stabil. Pertimbangkan stratified cross-validation pada data pengembangan, lalu pertahankan test set terpisah jika jumlah data memungkinkan.

### 7.2 Stratifikasi

Gunakan `stratify` agar proporsi kelas relatif terjaga di setiap split, terutama jika kelas tidak seimbang.

### 7.3 Hindari data leakage

- Jangan melakukan `fit` TF-IDF pada seluruh dataset sebelum split.
- Jangan menggunakan test set untuk memilih parameter atau model.
- Letakkan vectorizer dan classifier dalam satu `Pipeline`.
- Lakukan deduplikasi dan keputusan preprocessing dengan aturan yang konsisten dan terdokumentasi.
- Jika data memiliki dimensi waktu atau kelompok sumber, pertimbangkan split berbasis waktu/grup.

## 8. Metrik Evaluasi

Gunakan beberapa metrik, bukan accuracy saja.

### 8.1 Confusion matrix

Untuk kelas positif `Emergency`:

| | Prediksi Emergency | Prediksi Non-Emergency |
|---|---:|---:|
| Aktual Emergency | True Positive (TP) | False Negative (FN) |
| Aktual Non-Emergency | False Positive (FP) | True Negative (TN) |

- **False Negative:** tweet darurat diprediksi bukan darurat.
- **False Positive:** tweet bukan darurat diprediksi darurat.

Dalam konteks bantuan bencana, false negative perlu dianalisis secara serius karena berpotensi membuat laporan penting tidak terdeteksi. Namun, false positive juga perlu dikendalikan agar operator tidak kewalahan.

### 8.2 Accuracy

Proporsi prediksi yang benar dari seluruh data.

\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\]

### 8.3 Precision

Dari semua tweet yang diprediksi `Emergency`, berapa proporsi yang benar-benar `Emergency`?

\[
Precision = \frac{TP}{TP + FP}
\]

### 8.4 Recall

Dari semua tweet yang benar-benar `Emergency`, berapa proporsi yang berhasil ditemukan model?

\[
Recall = \frac{TP}{TP + FN}
\]

### 8.5 F1-score

Rata-rata harmonik precision dan recall.

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

Laporkan precision, recall, dan F1 per kelas. Sertakan **macro F1** jika kedua kelas sama-sama penting, serta recall kelas `Emergency`. Untuk data tidak seimbang, accuracy saja dapat menyesatkan.

## 9. Menyiapkan Environment

Pasang library yang diperlukan:

```bash
pip install pandas scikit-learn joblib
```

Gunakan Python environment terpisah agar versi dependensi proyek terdokumentasi.

## 10. Kode Eksperimen Lengkap

Simpan kode berikut sebagai `experiment.py`. Kode membaca `data/tweets.csv`, membuat split train/validation/test, membandingkan TF-IDF + Logistic Regression dan TF-IDF + Linear SVM, mencetak laporan evaluasi, dan menyimpan hasil ke folder `outputs/`.

Kode ini mengasumsikan kolom CSV bernama `text` dan `label`, dengan label `Emergency` serta `Non-Emergency`.

````python
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import joblib

RANDOM_STATE = 42
DATA_PATH = Path("data/tweets.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["Emergency", "Non-Emergency"]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}. "
            "Buat file CSV dengan kolom text dan label."
        )

    df = pd.read_csv(path)

    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Kolom wajib: {sorted(required)}")

    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df[df["label"].isin(LABELS)]

    # Hapus duplikat persis. Untuk riset, dokumentasikan kebijakan deduplikasi.
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    if df["label"].nunique() != 2:
        raise ValueError("Dataset harus memiliki kedua label: " + str(LABELS))

    counts = df["label"].value_counts()
    if counts.min() < 3:
        raise ValueError(
            "Setiap kelas membutuhkan lebih banyak contoh untuk split "
            "train/validation/test. Dataset saat ini terlalu kecil."
        )

    return df


def make_model(classifier):
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_df=1.0,
                sublinear_tf=True,
            ),
        ),
        ("classifier", classifier),
    ])


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=LABELS,
        zero_division=0,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=LABELS,
        target_names=LABELS,
        zero_division=0,
        digits=4,
    )
    matrix = confusion_matrix(y_test, predictions, labels=LABELS)

    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")
    print(f"Accuracy: {accuracy:.4f}")
    print(report)
    print("Confusion matrix (baris=aktual, kolom=prediksi):")
    print(pd.DataFrame(
        matrix,
        index=[f"Aktual {label}" for label in LABELS],
        columns=[f"Prediksi {label}" for label in LABELS],
    ))

    safe_name = name.lower().replace(" ", "_").replace("+", "plus")
    (OUTPUT_DIR / f"classification_report_{safe_name}.txt").write_text(
        f"{name}\n\nAccuracy: {accuracy:.4f}\n\n{report}\n"
        f"Confusion matrix:\n{matrix}\n",
        encoding="utf-8",
    )
    joblib.dump(model, OUTPUT_DIR / f"{safe_name}.joblib")

    return {
        "model": name,
        "accuracy": accuracy,
        "emergency_precision": precision[LABELS.index("Emergency")],
        "emergency_recall": recall[LABELS.index("Emergency")],
        "emergency_f1": f1[LABELS.index("Emergency")],
        "macro_f1": f1.mean(),
    }


def main():
    df = load_data(DATA_PATH)

    print("Jumlah data setelah pemeriksaan:", len(df))
    print("\nDistribusi label:")
    print(df["label"].value_counts())

    # 70% train, 30% sementara untuk validation + test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        df["text"],
        df["label"],
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    # Membagi 30% sementara menjadi 15% validation dan 15% test.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print("\nUkuran split:")
    print(f"Train      : {len(X_train)}")
    print(f"Validation : {len(X_val)}")
    print(f"Test       : {len(X_test)}")

    # Pada baseline sederhana ini, validation disiapkan untuk pemilihan
    # konfigurasi lanjutan. Test digunakan untuk perbandingan akhir.
    models = {
        "TF-IDF + Logistic Regression": make_model(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "TF-IDF + Linear SVM": make_model(
            LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)
        ),
    }

    results = []
    for name, model in models.items():
        results.append(
            evaluate_model(name, model, X_train, y_train, X_test, y_test)
        )

    results_df = pd.DataFrame(results).sort_values(
        "macro_f1", ascending=False
    )
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    print("\nPerbandingan model:")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nHasil disimpan di: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
````

### Catatan metodologis tentang kode

- `validation` dibuat, tetapi belum digunakan untuk tuning parameter pada contoh ini. Pada eksperimen final, gunakan validation atau cross-validation untuk memilih konfigurasi, lalu evaluasi model terpilih pada test set.
- `class_weight="balanced"` membantu mengakomodasi ketidakseimbangan kelas, tetapi tidak otomatis menjamin hasil lebih baik. Bandingkan dengan konfigurasi tanpa class weight jika relevan.
- Karena `TfidfVectorizer` berada di dalam `Pipeline`, vocabulary dan bobot IDF dipelajari hanya dari data train.
- Split acak cocok untuk latihan awal. Untuk data yang berurutan menurut waktu, data dari peristiwa yang sama, atau tweet yang berkelompok, split berbasis waktu/grup mungkin lebih sesuai.
- Jika jumlah data sangat kecil, stratifikasi pada split bertingkat dapat gagal. Gunakan dataset yang memadai atau desain validasi yang sesuai.

## 11. Menjalankan Eksperimen

Dari folder proyek, jalankan:

```bash
python experiment.py
```

Contoh tampilan output awal **bergantung pada dataset Anda**:

```text
Jumlah data setelah pemeriksaan: 2000

Distribusi label:
Non-Emergency    1200
Emergency         800

Ukuran split:
Train      : 1400
Validation : 300
Test       : 300
```

Angka di atas hanya ilustrasi untuk dataset dengan 2.000 baris valid. Program akan menampilkan hasil evaluasi aktual setelah model dilatih.

## 12. Contoh Output Evaluasi (Ilustratif)

Berikut contoh format hasil, **bukan hasil eksperimen yang benar-benar dijalankan pada dataset penelitian**:

```text
============================================================
TF-IDF + Logistic Regression
============================================================
Accuracy: 0.8600

              precision    recall  f1-score   support

   Emergency     0.8200    0.7800    0.7995       100
Non-Emergency    0.8800    0.9000    0.8899       200

    accuracy                         0.8600       300
   macro avg     0.8500    0.8400    0.8447       300
weighted avg     0.8600    0.8600    0.8598       300
```

Contoh tabel perbandingan (angka ilustratif):

| Model | Accuracy | Precision Emergency | Recall Emergency | F1 Emergency | Macro F1 |
|---|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.8600 | 0.8200 | 0.7800 | 0.7995 | 0.8447 |
| TF-IDF + Linear SVM | 0.8733 | 0.8400 | 0.8000 | 0.8195 | 0.8580 |

Angka tersebut hanya membantu memahami cara membaca tabel. Jangan menyalinnya sebagai hasil penelitian. Isi tabel paper/jurnal harus berasal dari eksperimen yang benar-benar dijalankan dan dapat direproduksi.

## 13. Membaca Hasil Eksperimen

Saat membandingkan model, jawab pertanyaan berikut:

1. Model mana yang memperoleh macro F1 tertinggi?
2. Berapa recall untuk kelas `Emergency`?
3. Apakah peningkatan recall disertai kenaikan false positive?
4. Apakah performa kedua kelas seimbang?
5. Apakah hasil stabil jika random seed atau fold cross-validation berubah?
6. Apakah ada indikasi kebocoran data atau duplikasi lintas split?

Jangan memilih model hanya berdasarkan accuracy. Pilih metrik utama berdasarkan kebutuhan penelitian dan jelaskan trade-off-nya.

## 14. Analisis Kesalahan

Simpan prediksi pada test set untuk mengidentifikasi:

- **False negative:** laporan bantuan darurat yang terlewat.
- **False positive:** tweet informasi umum yang dianggap darurat.
- **Ambiguitas konteks:** tweet terlalu singkat atau tidak jelas.
- **Bahasa informal:** singkatan, typo, bahasa daerah, atau pengulangan huruf.
- **Konteks lokasi:** lokasi disebutkan tanpa permintaan bantuan, atau permintaan bantuan tanpa lokasi.

Analisis kesalahan harus dilakukan dengan melihat teks dan label referensinya, bukan sekadar mengandalkan angka agregat. Catat pola kesalahan dan dampaknya pada rancangan sistem.

## 15. Eksperimen Lanjutan

Setelah baseline berjalan, pertimbangkan eksperimen terkontrol berikut.

### 15.1 Perbandingan algoritma

- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- TF-IDF + Multinomial Naive Bayes (opsional)

### 15.2 Ablation preprocessing

Bandingkan beberapa pilihan preprocessing, misalnya:
- teks minimal dibersihkan;
- normalisasi slang;
- penghapusan URL/mention;
- unigram dibandingkan unigram + bigram.

Ubah satu faktor pada satu waktu agar pengaruhnya lebih mudah ditafsirkan. Untuk model berbasis Transformer nanti, jangan otomatis menerapkan stemming atau penghapusan stopword; preprocessing harus disesuaikan dengan tokenizer dan model.

### 15.3 Tuning parameter

Contoh parameter yang bisa diuji:
- `ngram_range`
- `min_df`
- `max_features`
- `C` untuk Logistic Regression atau Linear SVM
- `class_weight`

Gunakan validation set atau cross-validation. Jangan memilih parameter berdasarkan test set.

### 15.4 Baseline menuju tesis IndoBERT

Pada tahap tesis, pertahankan definisi label, kebijakan deduplikasi, dan test set yang sama jika desain eksperimen memungkinkan. Bandingkan baseline klasik dengan fine-tuned IndoBERT secara adil.

| Aspek | Baseline klasik | Eksperimen lanjutan |
|---|---|---|
| Representasi | TF-IDF | Tokenizer dan contextual representation IndoBERT |
| Model | Logistic Regression / Linear SVM | IndoBERT fine-tuned |
| Data evaluasi | Test set yang ditetapkan | Test set yang sama jika memungkinkan |
| Metrik | Accuracy, precision, recall, F1 | Metrik yang sama |
| Fokus | Baseline yang kuat dan sederhana | Apakah konteks Transformer memberi peningkatan yang bermakna? |

## 16. Reproducibility dan Pelaporan

Catat untuk setiap eksperimen:

| Informasi | Yang perlu dicatat |
|---|---|
| Dataset | Sumber, tanggal pengumpulan, jumlah data |
| Label | Pedoman anotasi dan proses pemeriksaan |
| Split | Rasio, random seed, strategi split |
| Preprocessing | Langkah yang diterapkan |
| TF-IDF | Parameter vectorizer |
| Model | Algoritma dan parameter |
| Environment | Versi Python dan library |
| Evaluasi | Metrik per kelas dan confusion matrix |
| Artefak | Model tersimpan, laporan, dan kode |
| Keterbatasan | Bias data, kelas minoritas, domain bencana |

Simpan versi dataset yang digunakan. Jangan mengubah test set setelah melihat hasil tanpa mendokumentasikan perubahan tersebut.

## 17. Template Tabel Hasil Penelitian

Gunakan tabel berikut dalam catatan eksperimen dan isi dengan hasil aktual:

| Model | Accuracy | Precision (Emergency) | Recall (Emergency) | F1 (Emergency) | Macro F1 |
|---|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | — | — | — | — | — |
| TF-IDF + Linear SVM | — | — | — | — | — |
| TF-IDF + Multinomial Naive Bayes (opsional) | — | — | — | — | — |
| IndoBERT fine-tuned (tahap tesis) | — | — | — | — | — |

## 18. Checklist Sebelum Menyatakan Eksperimen Selesai

- [ ] Pertanyaan eksperimen telah ditentukan.
- [ ] Definisi `Emergency` dan `Non-Emergency` terdokumentasi.
- [ ] Data kosong dan duplikat diperiksa.
- [ ] Distribusi label dilaporkan.
- [ ] Split data dibuat tanpa kebocoran.
- [ ] Pipeline TF-IDF dilatih hanya pada data train.
- [ ] Semua model memakai test set yang sama.
- [ ] Precision, recall, F1, dan confusion matrix dilaporkan.
- [ ] Recall kelas `Emergency` dianalisis.
- [ ] Kesalahan model ditinjau.
- [ ] Versi kode, parameter, dan dataset dicatat.
- [ ] Angka yang dilaporkan berasal dari hasil aktual, bukan contoh ilustratif.

## 19. Ringkasan

Eksperimen penelitian yang baik bukan tentang mendapatkan angka tertinggi semata. Eksperimen harus menjawab pertanyaan penelitian, menggunakan perbandingan yang adil, mencegah data leakage, melaporkan metrik yang relevan, dan dapat direproduksi.

Untuk tahap proyek ini, mulailah dengan **TF-IDF + Logistic Regression** dan **TF-IDF + Linear SVM**. Setelah baseline dan kualitas dataset teruji, hasilnya dapat menjadi pembanding yang kuat untuk eksperimen fine-tuning IndoBERT pada tesis.

**Target akhir materi:** menghasilkan laporan eksperimen yang berisi deskripsi dataset, metode, konfigurasi, metrik evaluasi, analisis kesalahan, dan kesimpulan berbasis bukti.
