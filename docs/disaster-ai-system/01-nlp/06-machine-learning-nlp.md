# Machine Learning untuk NLP (Natural Language Processing)

## Tujuan Pembelajaran

Setelah mempelajari materi ini, Anda diharapkan mampu:

1. Menjelaskan alur machine learning untuk klasifikasi teks.
2. Mengubah teks menjadi fitur numerik menggunakan TF-IDF.
3. Membagi dataset menjadi data latih dan data uji.
4. Melatih model klasifikasi teks menggunakan Logistic Regression dan Support Vector Machine (SVM).
5. Mengevaluasi model menggunakan confusion matrix, precision, recall, dan F1-score.
6. Membuat pipeline yang menggabungkan TF-IDF dan model klasifikasi.
7. Memahami baseline eksperimen untuk penelitian deteksi tweet permintaan bantuan darurat saat bencana.

## 1. Pengantar Machine Learning untuk NLP

Natural Language Processing (NLP) adalah bidang yang membuat komputer dapat memproses dan menganalisis bahasa manusia. Teks tidak dapat langsung diproses oleh sebagian besar algoritma machine learning klasik. Teks perlu diubah menjadi representasi numerik terlebih dahulu.

Contoh tugas NLP yang dapat diselesaikan menggunakan machine learning:

- Klasifikasi spam dan bukan spam.
- Analisis sentimen.
- Klasifikasi topik berita.
- Deteksi ujaran kebencian.
- Klasifikasi tweet permintaan bantuan darurat dan bukan permintaan bantuan darurat.

Pada materi ini, kita menggunakan **klasifikasi teks (text classification)** sebagai contoh utama.

### 1.1 Contoh masalah klasifikasi

Misalnya, sistem perlu membedakan tweet yang mengindikasikan permintaan bantuan darurat dan tweet yang hanya menyampaikan informasi umum.

| Teks | Label contoh |
|---|---|
| Tolong bantu kami, air banjir masuk ke rumah | Emergency |
| Kami membutuhkan makanan dan air bersih setelah banjir | Emergency |
| Hujan turun sejak pagi di wilayah ini | Non-Emergency |
| Berita tentang banjir disiarkan malam ini | Non-Emergency |

Contoh di atas hanya ilustrasi. Dalam penelitian nyata, label harus ditetapkan berdasarkan pedoman anotasi yang jelas. Kata seperti *banjir* tidak otomatis berarti sebuah tweet merupakan permintaan bantuan darurat.

### 1.2 Alur umum machine learning untuk NLP

```text
Kumpulan Teks Berlabel
        |
        v
Pemeriksaan dan Persiapan Data
        |
        v
Pembagian Train dan Test
        |
        v
Text Vectorization (misalnya TF-IDF)
        |
        v
Pelatihan Model
        |
        v
Prediksi pada Data Test
        |
        v
Evaluasi
```

**Catatan:** untuk menghindari data leakage, vectorizer harus dipelajari hanya dari data latih. Cara yang aman adalah menggunakan `Pipeline` dari scikit-learn, yang akan dibahas kemudian.

## 2. Dataset dan Label

Dataset klasifikasi teks biasanya memiliki setidaknya dua kolom:

- `text`: teks yang akan diklasifikasikan.
- `label`: kategori target.

Contoh:

| text | label |
|---|---|
| Tolong bantu kami, rumah terendam banjir | Emergency |
| Kami kekurangan air bersih di pengungsian | Emergency |
| Hujan deras mengguyur kota sejak pagi | Non-Emergency |
| Pemerintah mengadakan konferensi pers tentang cuaca | Non-Emergency |

Untuk model biner, label dapat dikodekan sebagai `1` dan `0`, tetapi arti setiap angka harus didokumentasikan secara konsisten.

### 2.1 Contoh membuat dataset dengan Python

Kode berikut membuat dataset kecil untuk latihan. Dataset ini **sengaja dibuat sederhana dan sintetis**, bukan dataset penelitian atau bukti performa model di dunia nyata.

```python
import pandas as pd

data = {
    "text": [
        "Tolong bantu kami rumah terendam banjir",
        "Kami membutuhkan makanan di pengungsian",
        "Mohon kirim air bersih ke desa kami",
        "Ada warga terjebak dan butuh evakuasi",
        "Hujan turun sejak pagi di kota",
        "Berita banjir disiarkan malam ini",
        "Cuaca hari ini cukup cerah",
        "Pemerintah mengadakan rapat kebencanaan",
        "Kami kehabisan makanan setelah banjir",
        "Jalan menuju desa tertutup longsor"
    ],
    "label": [
        "Emergency",
        "Emergency",
        "Emergency",
        "Emergency",
        "Non-Emergency",
        "Non-Emergency",
        "Non-Emergency",
        "Non-Emergency",
        "Emergency",
        "Non-Emergency"
    ]
}

df = pd.DataFrame(data)

print(df.head())
print("\\nJumlah data:", len(df))
print("\\nDistribusi label:")
print(df["label"].value_counts())
```

Contoh output:

```text
                                          text          label
0  Tolong bantu kami rumah terendam banjir      Emergency
1  Kami membutuhkan makanan di pengungsian      Emergency
2  Mohon kirim air bersih ke desa kami          Emergency
3  Ada warga terjebak dan butuh evakuasi        Emergency
4  Hujan turun sejak pagi di kota               Non-Emergency

Jumlah data: 10

Distribusi label:
Emergency        5
Non-Emergency    5
Name: count, dtype: int64
```

Output tampilan dapat sedikit berbeda tergantung versi pandas.

## 3. Memahami Fitur dan Target

Dalam machine learning:

- **X (fitur)** adalah masukan yang digunakan model untuk membuat prediksi.
- **y (target)** adalah label yang ingin diprediksi.

Untuk dataset teks:

```python
X = df["text"]
y = df["label"]

print("Contoh X:")
print(X.iloc[0])

print("\\nContoh y:")
print(y.iloc[0])
```

Output:

```text
Contoh X:
Tolong bantu kami rumah terendam banjir

Contoh y:
Emergency
```

Pada tahap ini, `X` masih berupa teks. Model machine learning klasik membutuhkan fitur numerik, sehingga teks akan diubah menjadi vektor.

## 4. Membagi Dataset: Train dan Test

- **Training set** digunakan untuk mempelajari pola.
- **Test set** digunakan untuk mengevaluasi model pada data yang tidak digunakan selama pelatihan.

Contoh pembagian 80% data latih dan 20% data uji:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Jumlah data latih:", len(X_train))
print("Jumlah data uji:", len(X_test))
print("\\nDistribusi label data latih:")
print(y_train.value_counts())
print("\\nDistribusi label data uji:")
print(y_test.value_counts())
```

Karena contoh dataset hanya memiliki 10 baris, hasilnya adalah:

```text
Jumlah data latih: 8
Jumlah data uji: 2

Distribusi label data latih:
Emergency        4
Non-Emergency    4
Name: count, dtype: int64

Distribusi label data uji:
Emergency        1
Non-Emergency    1
Name: count, dtype: int64
```

`stratify=y` berusaha mempertahankan proporsi label pada setiap bagian. Pada dataset penelitian, gunakan data yang cukup dan pertimbangkan pemisahan berdasarkan waktu, sumber, atau peristiwa bila diperlukan agar evaluasi lebih realistis.

**Penting:** jangan menggunakan data test untuk memilih model, mengatur hyperparameter, atau menentukan preprocessing. Untuk pemilihan model, gunakan data validasi atau cross-validation pada data latih.

## 5. Mengubah Teks Menjadi Angka dengan TF-IDF

### 5.1 Mengapa teks harus diubah?

Algoritma seperti Logistic Regression dan SVM bekerja dengan fitur numerik. TF-IDF mengubah dokumen menjadi vektor berdasarkan bobot kata.

TF-IDF merupakan singkatan dari:

- **TF (Term Frequency):** ukuran frekuensi suatu istilah dalam dokumen.
- **IDF (Inverse Document Frequency):** ukuran seberapa jarang istilah muncul di seluruh dokumen.
- **TF-IDF:** menggabungkan kedua ukuran tersebut untuk memberi bobot pada istilah.

Secara intuitif, kata yang muncul pada banyak dokumen dapat memiliki bobot pembeda yang lebih kecil daripada kata yang lebih khas pada sebagian dokumen.

### 5.2 Contoh TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

contoh_teks = [
    "butuh bantuan banjir",
    "bantuan air bersih",
    "cuaca cerah hari ini"
]

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(contoh_teks)

print("Vocabulary:")
print(vectorizer.get_feature_names_out())

print("\\nBentuk matriks TF-IDF:")
print(X_tfidf.shape)

print("\\nMatriks TF-IDF:")
print(X_tfidf.toarray().round(3))
```

Contoh output:

```text
Vocabulary:
['air' 'bantuan' 'banjir' 'bersih' 'butuh' 'cerah' 'cuaca' 'hari' 'ini']

Bentuk matriks TF-IDF:
(3, 9)

Matriks TF-IDF:
[[0.    0.469 0.619 0.    0.619 0.    0.    0.    0.   ]
 [0.577 0.577 0.    0.577 0.    0.    0.    0.    0.   ]
 [0.    0.    0.    0.    0.    0.577 0.577 0.577 0.577]]
```

**Catatan:** nilai TF-IDF bisa berbeda jika parameter atau versi library berbeda. Perhatikan bahwa matriks memiliki 3 baris (dokumen) dan 9 kolom (fitur kata). Dalam penggunaan sebenarnya, matriks TF-IDF biasanya berupa sparse matrix karena sebagian besar nilainya nol.

### 5.3 Apa yang dipelajari vectorizer?

`fit()` mempelajari vocabulary dan statistik yang diperlukan dari data. `transform()` menggunakan hasil pembelajaran tersebut untuk mengubah teks menjadi vektor.

- `fit_transform(X_train)`: pelajari vocabulary dari data latih dan ubah data latih.
- `transform(X_test)`: gunakan vocabulary yang sudah dipelajari untuk mengubah data uji.
- Hindari `fit_transform()` pada seluruh dataset sebelum train-test split karena dapat menyebabkan **data leakage**.

## 6. Model Pertama: Logistic Regression

Walaupun namanya mengandung kata *regression*, Logistic Regression sering digunakan untuk klasifikasi.

Model mempelajari hubungan antara fitur teks dan label, lalu menghasilkan skor atau probabilitas kelas (bergantung pada konfigurasi).

### 6.1 Contoh implementasi dasar

Contoh berikut sengaja ditulis untuk memperlihatkan langkah TF-IDF dan model secara terpisah. Dalam proyek nyata, lebih aman menggunakan `Pipeline` pada bagian 9.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("Prediksi:", y_pred.tolist())
print("Label aktual:", y_test.tolist())
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
```

Contoh output yang mungkin muncul pada dataset sintetis ini:

```text
Prediksi: ['Emergency', 'Non-Emergency']
Label aktual: ['Emergency', 'Non-Emergency']
Accuracy: 1.0
```

Output prediksi dan accuracy dapat berubah jika dataset atau versi library berubah. Karena data contoh sangat kecil dan dibuat secara sederhana, hasil 1.0 **tidak menunjukkan bahwa model sudah bagus untuk tweet nyata**.

### 6.2 Cara membaca hasil

Jika accuracy adalah `1.0`, artinya model memprediksi dengan benar seluruh contoh pada test set yang sangat kecil ini. Jika test set hanya berisi dua data, satu prediksi yang salah saja akan mengubah accuracy secara drastis.

Karena itu, jangan menilai model hanya dari satu angka atau dataset kecil.

## 7. Model Kedua: Support Vector Machine (SVM)

SVM mencari batas pemisah yang membedakan kelas dalam ruang fitur. Untuk klasifikasi teks dengan TF-IDF, `LinearSVC` sering menjadi baseline yang kuat dan efisien.

### 7.1 Implementasi

```python
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

svm_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", LinearSVC(random_state=42))
])

svm_pipeline.fit(X_train, y_train)
svm_pred = svm_pipeline.predict(X_test)

print(classification_report(
    y_test,
    svm_pred,
    labels=["Emergency", "Non-Emergency"],
    zero_division=0
))
```

Contoh format output:

```text
               precision    recall  f1-score   support

    Emergency       1.00      1.00      1.00         1
Non-Emergency       1.00      1.00      1.00         1

     accuracy                           1.00         2
    macro avg       1.00      1.00      1.00         2
 weighted avg       1.00      1.00      1.00         2
```

Ini adalah contoh format output; angka sebenarnya mengikuti prediksi model yang dijalankan. Dengan satu contoh per kelas pada test set, precision, recall, dan F1-score sangat tidak stabil.

## 8. Evaluasi Model

### 8.1 Confusion Matrix

Confusion matrix menunjukkan jumlah prediksi benar dan salah untuk setiap kelas.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

labels = ["Emergency", "Non-Emergency"]
cm = confusion_matrix(y_test, svm_pred, labels=labels)

print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)
display.plot()
plt.tight_layout()
plt.show()
```

Jika kedua prediksi benar, output matriksnya:

```text
[[1 0]
 [0 1]]
```

Urutan baris dan kolom mengikuti `labels` yang ditentukan. Baris mewakili label aktual dan kolom mewakili prediksi.

### 8.2 Accuracy

Accuracy adalah proporsi seluruh prediksi yang benar.

\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\]

Accuracy berguna, tetapi dapat menyesatkan jika kelas tidak seimbang.

### 8.3 Precision

Precision menjawab:

> Dari semua data yang diprediksi sebagai Emergency, berapa banyak yang benar-benar Emergency?

\[
Precision = \frac{TP}{TP + FP}
\]

### 8.4 Recall

Recall menjawab:

> Dari semua data yang sebenarnya Emergency, berapa banyak yang berhasil ditemukan model?

\[
Recall = \frac{TP}{TP + FN}
\]

Untuk sistem penyaringan permintaan bantuan darurat, **recall kelas Emergency perlu diperhatikan**, sebab false negative berarti laporan yang berpotensi darurat tidak terdeteksi. Namun, precision juga tetap penting agar sistem tidak membanjiri petugas dengan terlalu banyak peringatan keliru.

### 8.5 F1-score

F1-score merupakan rata-rata harmonik precision dan recall.

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

### 8.6 Laporan metrik

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("Accuracy:", round(accuracy_score(y_test, svm_pred), 3))
print(
    "Emergency precision:",
    round(precision_score(
        y_test, svm_pred,
        pos_label="Emergency",
        zero_division=0
    ), 3)
)
print(
    "Emergency recall:",
    round(recall_score(
        y_test, svm_pred,
        pos_label="Emergency",
        zero_division=0
    ), 3)
)
print(
    "Emergency F1:",
    round(f1_score(
        y_test, svm_pred,
        pos_label="Emergency",
        zero_division=0
    ), 3)
)
```

Contoh output jika seluruh prediksi benar:

```text
Accuracy: 1.0
Emergency precision: 1.0
Emergency recall: 1.0
Emergency F1: 1.0
```

Sekali lagi, hasil ini hanya ilustrasi pada dataset latihan yang sangat kecil.

## 9. Pipeline: Cara yang Direkomendasikan

`Pipeline` menyatukan vectorizer dan model. Keuntungannya:

1. TF-IDF dipelajari hanya dari data yang diberikan kepada `fit()`.
2. Transformasi yang sama diterapkan saat prediksi.
3. Kode lebih ringkas dan mengurangi risiko data leakage.
4. Pipeline mudah digunakan bersama cross-validation dan pencarian hyperparameter.

### 9.1 Membuat pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

lr_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

lr_pipeline.fit(X_train, y_train)

prediksi = lr_pipeline.predict(X_test)

print("Prediksi:", prediksi.tolist())
```

Contoh output:

```text
Prediksi: ['Emergency', 'Non-Emergency']
```

Prediksi aktual dapat berbeda. `ngram_range=(1, 2)` berarti vectorizer menggunakan unigram (satu kata) dan bigram (dua kata berurutan). Bigram dapat menangkap frasa seperti “air bersih” atau “butuh bantuan”.

### 9.2 Prediksi teks baru

```python
teks_baru = [
    "Tolong kirim bantuan, warga terjebak banjir",
    "Cuaca cerah dan matahari bersinar"
]

hasil = lr_pipeline.predict(teks_baru)

for teks, label in zip(teks_baru, hasil):
    print(f"Teks: {teks}")
    print(f"Prediksi: {label}")
    print()
```

Contoh output:

```text
Teks: Tolong kirim bantuan, warga terjebak banjir
Prediksi: Emergency

Teks: Cuaca cerah dan matahari bersinar
Prediksi: Non-Emergency
```

Output tersebut hanya ilustrasi; model yang dilatih pada dataset kecil belum layak digunakan untuk keputusan operasional.

## 10. Preprocessing untuk TF-IDF dan BERT Tidak Selalu Sama

Untuk TF-IDF, normalisasi seperti lowercase, penghapusan URL, atau normalisasi slang dapat membantu—tetapi harus diuji, bukan diasumsikan selalu lebih baik.

Untuk BERT/IndoBERT, hindari melakukan pembersihan agresif secara otomatis. Penghapusan tanda baca, emoji, negasi, kata berulang, atau kata-kata informal bisa menghilangkan sinyal penting. Umumnya, gunakan tokenizer bawaan model dan pertahankan teks sedekat mungkin dengan bentuk aslinya, kecuali eksperimen menunjukkan manfaat preprocessing tambahan.

Prinsip yang disarankan:

- Simpan teks asli.
- Buat preprocessing sebagai langkah yang terdokumentasi.
- Bandingkan hasil dengan dan tanpa langkah tertentu.
- Jangan menghapus kata negasi seperti “tidak” tanpa alasan yang kuat.
- Jangan melakukan stemming untuk IndoBERT secara otomatis.

## 11. Cross-Validation dan Pemilihan Model

Satu pembagian train-test dapat menghasilkan estimasi yang tidak stabil, khususnya jika dataset kecil. Cross-validation membagi data latih menjadi beberapa lipatan untuk menilai model pada beberapa pembagian.

Contoh:

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

cv_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("model", LinearSVC(random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    cv_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="f1_macro"
)

print("Skor setiap fold:", scores.round(3))
print("Rata-rata macro F1:", round(scores.mean(), 3))
```

**Penting:** contoh dataset 10 baris pada materi ini terlalu kecil untuk 5-fold cross-validation yang andal. Kode tersebut menunjukkan pola penggunaan; untuk menjalankannya, gunakan dataset yang lebih besar dengan jumlah contoh yang memadai pada setiap kelas. Jangan menganggap skor dari dataset mini sebagai hasil penelitian.

Dalam penelitian, pilih metrik sesuai tujuan. Macro F1 memberi bobot yang sama pada setiap kelas dan berguna ketika performa kelas minoritas penting.

## 12. Menangani Ketidakseimbangan Kelas

Dataset bencana mungkin memiliki jauh lebih banyak tweet informasi umum daripada permintaan bantuan darurat, atau sebaliknya bergantung pada cara pengumpulan data.

Contoh distribusi:

| Label | Jumlah |
|---|---:|
| Emergency | 1.000 |
| Non-Emergency | 9.000 |

Model yang selalu menebak `Non-Emergency` dapat terlihat memiliki accuracy tinggi, tetapi gagal mendeteksi kelas yang penting.

Langkah yang bisa diuji:

- Periksa distribusi label.
- Laporkan precision, recall, F1 per kelas, dan macro F1.
- Coba `class_weight="balanced"` pada model yang mendukungnya.
- Evaluasi threshold jika model menghasilkan probabilitas atau skor yang sesuai.
- Lakukan oversampling atau undersampling hanya pada data latih, bukan sebelum pemisahan data.
- Pastikan data test tetap mewakili kondisi penggunaan yang ingin dievaluasi.

Contoh SVM dengan class weight:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

balanced_svm = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("model", LinearSVC(class_weight="balanced", random_state=42))
])

balanced_svm.fit(X_train, y_train)
```

`class_weight="balanced"` menyesuaikan bobot kelas berdasarkan frekuensi label dalam data latih. Penggunaan ini tidak otomatis menjamin hasil lebih baik; ukur efeknya dengan evaluasi yang sesuai.

## 13. Baseline untuk Penelitian Deteksi Permintaan Bantuan Darurat

Sebelum mengklaim IndoBERT lebih baik, buat baseline yang dapat direproduksi.

Rancangan eksperimen awal:

| Eksperimen | Representasi | Model |
|---|---|---|
| 1 | TF-IDF unigram | Logistic Regression |
| 2 | TF-IDF unigram + bigram | Linear SVM |
| 3 | TF-IDF unigram + bigram | Logistic Regression |
| 4 | Tokenizer IndoBERT | Fine-tuning IndoBERT |

Gunakan pemisahan data yang sama dan adil jika memungkinkan. Tetapkan test set sebelum memilih model dan jangan menggunakannya untuk tuning. Untuk data media sosial, pertimbangkan duplikasi tweet, retweet, kebocoran antar-sumber, dan pemisahan berdasarkan waktu atau kejadian bencana jika relevan.

Laporkan sekurang-kurangnya:

- Jumlah data dan distribusi kelas.
- Cara pengumpulan data dan pedoman anotasi.
- Strategi train/validation/test split.
- Parameter preprocessing dan model.
- Precision, recall, F1 per kelas, macro F1, dan confusion matrix.
- Keterbatasan data dan risiko kesalahan klasifikasi.

Jangan menganggap peningkatan metrik otomatis merupakan novelty penelitian. Kontribusi juga dapat berasal dari kualitas dataset, pedoman anotasi, desain evaluasi, analisis kesalahan, atau manfaat sistem dalam konteks kebencanaan.

## 14. Latihan Mandiri

Kerjakan latihan berikut secara berurutan.

### Latihan 1 — Pemeriksaan data

1. Tampilkan 10 baris pertama dataset.
2. Hitung jumlah data.
3. Hitung distribusi setiap label.
4. Periksa apakah ada teks kosong atau duplikat.

### Latihan 2 — TF-IDF

1. Buat `TfidfVectorizer`.
2. Fit hanya pada `X_train`.
3. Tampilkan 10 fitur pertama.
4. Periksa bentuk matriks hasil transformasi.

### Latihan 3 — Perbandingan model

Bandingkan Logistic Regression dan Linear SVM menggunakan pipeline. Catat hasil accuracy, precision, recall, dan F1-score.

### Latihan 4 — Analisis kesalahan

Temukan contoh false positive dan false negative. Periksa apakah kesalahan berkaitan dengan kata ambigu, konteks, negasi, atau kualitas label.

### Latihan 5 — Eksperimen preprocessing

Bandingkan hasil model dengan beberapa kondisi:

- Teks asli dengan lowercase yang dilakukan vectorizer.
- Pembersihan URL dan mention.
- Normalisasi istilah informal yang memiliki kamus terdokumentasi.

Gunakan pembagian data yang sama dan jangan memilih konfigurasi berdasarkan test set.

## 15. Checklist Penguasaan

Gunakan checklist ini sebelum melanjutkan ke materi IndoBERT.

- [ ] Saya memahami perbedaan fitur `X` dan target `y`.
- [ ] Saya dapat membagi data menjadi train dan test.
- [ ] Saya memahami konsep TF-IDF dan sparse matrix.
- [ ] Saya dapat melatih Logistic Regression untuk klasifikasi teks.
- [ ] Saya dapat melatih Linear SVM untuk klasifikasi teks.
- [ ] Saya dapat menggunakan `Pipeline`.
- [ ] Saya dapat membaca confusion matrix.
- [ ] Saya memahami precision, recall, dan F1-score.
- [ ] Saya memahami data leakage dan alasan vectorizer hanya di-fit pada data latih.
- [ ] Saya dapat menjelaskan mengapa dataset mini tidak cukup untuk menarik kesimpulan penelitian.
- [ ] Saya dapat membuat baseline yang akan dibandingkan dengan IndoBERT.

Jika sebagian besar checklist sudah terpenuhi, langkah berikutnya adalah mempelajari **Word Embedding dan dasar Deep Learning**, lalu Attention, Transformer, BERT, dan fine-tuning IndoBERT.

## 16. Ringkasan

Alur dasar machine learning untuk NLP adalah:

```text
Teks Berlabel
    ↓
Pemisahan Data
    ↓
TF-IDF
    ↓
Model Klasifikasi
    ↓
Prediksi
    ↓
Evaluasi
```

Hal terpenting untuk diingat:

1. Teks perlu direpresentasikan secara numerik untuk model ML klasik.
2. TF-IDF + Logistic Regression atau Linear SVM adalah baseline yang layak dicoba.
3. Gunakan pipeline untuk mengurangi risiko data leakage.
4. Evaluasi jangan hanya berfokus pada accuracy; perhatikan recall kelas Emergency dan macro F1.
5. Dataset sintetis pada materi ini hanya untuk belajar sintaks, bukan untuk membuktikan kemampuan model di dunia nyata.
6. Baseline yang kuat membuat perbandingan dengan IndoBERT lebih bermakna.
