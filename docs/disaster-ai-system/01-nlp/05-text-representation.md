# Text Representation & TF-IDF dalam NLP (Natural Language Processing)

## 1. Tujuan Pembelajaran

Setelah mempelajari materi ini, Anda diharapkan mampu:

1. Menjelaskan mengapa teks perlu diubah menjadi representasi numerik.
2. Memahami token, vocabulary, document, corpus, dan feature.
3. Mengubah teks menjadi angka menggunakan One-Hot Encoding.
4. Memahami dan menerapkan Bag of Words (BoW).
5. Menggunakan `CountVectorizer` dari scikit-learn.
6. Memahami Term Frequency (TF), Inverse Document Frequency (IDF), dan TF-IDF.
7. Menggunakan `TfidfVectorizer` untuk membentuk fitur teks.
8. Membaca hasil representasi teks dan mengenali keterbatasannya.
9. Menyiapkan baseline representasi teks untuk penelitian klasifikasi tweet permintaan bantuan darurat.

## 2. Apa Itu Text Representation?

**Text representation** adalah proses mengubah teks menjadi bentuk numerik agar dapat diproses oleh algoritma Machine Learning. Model seperti Logistic Regression, Naive Bayes, dan Support Vector Machine (SVM) bekerja dengan fitur numerik, bukan dengan kalimat mentah secara langsung.

Contoh teks:

```text
"Tolong bantu korban banjir"
```

Model Machine Learning tidak otomatis memahami makna kalimat tersebut seperti manusia. Teks perlu diubah menjadi fitur numerik terlebih dahulu.

Alur umumnya:

```text
Teks mentah
    ↓
Preprocessing (sesuai kebutuhan)
    ↓
Text representation
    ↓
Vektor numerik
    ↓
Model Machine Learning
    ↓
Prediksi
```

Pada materi ini, metode yang dipelajari adalah:

1. One-Hot Encoding
2. Bag of Words (BoW)
3. CountVectorizer
4. Term Frequency (TF)
5. Inverse Document Frequency (IDF)
6. TF-IDF

Metode seperti Word2Vec, FastText, dan contextual embedding dari BERT/IndoBERT dibahas pada tahap pembelajaran berikutnya.

## 3. Istilah Dasar

### 3.1 Document

**Document** adalah satu unit teks yang akan diproses. Dalam klasifikasi tweet, satu tweet biasanya diperlakukan sebagai satu document.

Contoh:

```text
Document 1: "tolong bantu kami"
Document 2: "banjir merendam rumah"
Document 3: "cuaca hari ini cerah"
```

### 3.2 Corpus

**Corpus** adalah kumpulan document yang digunakan untuk analisis atau pemodelan bahasa.

Contoh corpus kecil:

```python
documents = [
    "tolong bantu kami",
    "banjir merendam rumah",
    "cuaca hari ini cerah"
]

print("Jumlah dokumen:", len(documents))
```

Output:

```text
Jumlah dokumen: 3
```

### 3.3 Token

**Token** adalah unit hasil pemecahan teks. Token dapat berupa kata, subkata, atau unit lain, bergantung pada tokenizer yang digunakan.

Contoh:

```text
"Tolong bantu kami"
        ↓
["Tolong", "bantu", "kami"]
```

### 3.4 Vocabulary

**Vocabulary** adalah kumpulan token unik yang dikenali dari corpus oleh metode representasi tertentu.

Untuk corpus:

```text
"tolong bantu kami"
"bantu korban banjir"
```

Vocabulary sederhananya:

```text
["tolong", "bantu", "kami", "korban", "banjir"]
```

Urutan vocabulary dapat berbeda, bergantung pada implementasi atau pengaturan library. Untuk contoh manual pada materi ini, urutan ditentukan secara eksplisit agar mudah diikuti.

### 3.5 Feature dan Feature Vector

- **Feature** adalah satu atribut numerik yang mewakili informasi dari teks, misalnya keberadaan atau frekuensi kata.
- **Feature vector** adalah deretan angka yang mewakili satu document.
- **Feature matrix** adalah kumpulan feature vector untuk seluruh document. Baris biasanya mewakili document dan kolom mewakili fitur.

Jika vocabulary berisi lima kata, representasi BoW standar akan memiliki lima fitur untuk setiap document.

## 4. One-Hot Encoding

### 4.1 Konsep

One-Hot Encoding dapat digunakan untuk merepresentasikan satu token sebagai vektor yang hanya memiliki satu nilai `1`, sedangkan posisi lainnya bernilai `0`.

Misalnya vocabulary:

```text
["bantu", "banjir", "korban"]
```

Representasi:

| Token | Vektor |
|---|---|
| bantu | `[1, 0, 0]` |
| banjir | `[0, 1, 0]` |
| korban | `[0, 0, 1]` |

Setiap posisi mewakili satu token di dalam vocabulary.

### 4.2 Implementasi Python

```python
vocabulary = ["bantu", "banjir", "korban"]

def one_hot_encode(token, vocabulary):
    vector = [0] * len(vocabulary)

    if token in vocabulary:
        index = vocabulary.index(token)
        vector[index] = 1

    return vector

print("bantu :", one_hot_encode("bantu", vocabulary))
print("banjir:", one_hot_encode("banjir", vocabulary))
print("korban:", one_hot_encode("korban", vocabulary))
```

Output:

```text
bantu : [1, 0, 0]
banjir: [0, 1, 0]
korban: [0, 0, 1]
```

### 4.3 Kelebihan dan Kekurangan

**Kelebihan:**

- Sederhana dan mudah dipahami.
- Menunjukkan identitas token secara eksplisit.

**Kekurangan:**

- Vektor dapat menjadi sangat panjang jika vocabulary besar.
- Representasi ini tidak menunjukkan kemiripan makna antarkata.
- Pada bentuk dasarnya, setiap token hanya ditandai keberadaannya, bukan frekuensi pemakaiannya.

One-Hot Encoding berguna untuk memahami konsep representasi numerik, tetapi bukan satu-satunya atau selalu pilihan terbaik untuk klasifikasi teks.

## 5. Bag of Words (BoW)

### 5.1 Konsep

**Bag of Words (BoW)** merepresentasikan document berdasarkan jumlah kemunculan setiap kata dalam vocabulary. Metode ini mengabaikan urutan kata dan berfokus pada frekuensi token.

Contoh corpus:

```text
Document 1: "bantu korban banjir"
Document 2: "bantu korban"
Document 3: "banjir datang"
```

Vocabulary yang dipakai pada contoh:

```text
["bantu", "korban", "banjir", "datang"]
```

Matriks BoW:

| Document | bantu | korban | banjir | datang |
|---|---:|---:|---:|---:|
| Document 1 | 1 | 1 | 1 | 0 |
| Document 2 | 1 | 1 | 0 | 0 |
| Document 3 | 0 | 0 | 1 | 1 |

Vektor untuk Document 1 adalah `[1, 1, 1, 0]`.

### 5.2 Implementasi BoW secara Manual

```python
documents = [
    "bantu korban banjir",
    "bantu korban",
    "banjir datang"
]

vocabulary = ["bantu", "korban", "banjir", "datang"]

def bow_vectorize(document, vocabulary):
    tokens = document.split()
    return [tokens.count(word) for word in vocabulary]

for i, document in enumerate(documents, start=1):
    print(f"Document {i}: {bow_vectorize(document, vocabulary)}")
```

Output:

```text
Document 1: [1, 1, 1, 0]
Document 2: [1, 1, 0, 0]
Document 3: [0, 0, 1, 1]
```

Implementasi manual ini sengaja dibuat sederhana. `split()` hanya memisahkan berdasarkan spasi dan belum menangani tanda baca, variasi kata, atau tokenisasi yang lebih kompleks.

### 5.3 Kelebihan dan Kekurangan BoW

**Kelebihan:**

- Mudah diterapkan.
- Hasilnya relatif mudah dijelaskan.
- Dapat menjadi baseline untuk klasifikasi teks.

**Kekurangan:**

- Tidak mempertahankan urutan kata.
- Tidak memahami sinonim atau konteks.
- Kata yang sangat sering muncul dapat mendominasi representasi.
- Vocabulary besar dapat menghasilkan matriks sparse berdimensi tinggi.

Perhatikan bahwa kalimat `"korban membantu warga"` dan `"warga membantu korban"` dapat mempunyai vektor BoW yang sama meskipun susunan katanya berbeda.

## 6. CountVectorizer dari scikit-learn

### 6.1 Apa Itu CountVectorizer?

`CountVectorizer` adalah alat dari scikit-learn untuk mengubah kumpulan dokumen teks menjadi matriks jumlah kemunculan token. Secara default, alat ini melakukan ekstraksi token berbasis pola teks dan mengabaikan token yang hanya terdiri dari satu karakter.

### 6.2 Instalasi

Jika scikit-learn belum tersedia:

```bash
pip install scikit-learn
```

### 6.3 Contoh Kode

```python
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "bantu korban banjir",
    "bantu korban",
    "banjir datang"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

print("Vocabulary:")
print(vectorizer.get_feature_names_out())

print("\nMatriks BoW:")
print(X.toarray())
```

Output:

```text
Vocabulary:
['bantu' 'banjir' 'datang' 'korban']

Matriks BoW:
[[1 1 0 1]
 [1 0 0 1]
 [0 1 1 0]]
```

**Mengapa urutannya berbeda dari contoh manual?** `CountVectorizer` mengurutkan nama fitur secara alfabetis pada output ini. Oleh karena itu, kolomnya adalah `bantu`, `banjir`, `datang`, dan `korban`.

### 6.4 Memeriksa Vocabulary dan Vektor

```python
feature_names = vectorizer.get_feature_names_out()

for i, document in enumerate(documents):
    print(f"Document {i + 1}: {document}")
    print(dict(zip(feature_names, X.toarray()[i])))
```

Output:

```text
Document 1: bantu korban banjir
{'bantu': np.int64(1), 'banjir': np.int64(1), 'datang': np.int64(0), 'korban': np.int64(1)}
Document 2: bantu korban
{'bantu': np.int64(1), 'banjir': np.int64(0), 'datang': np.int64(0), 'korban': np.int64(1)}
Document 3: banjir datang
{'bantu': np.int64(0), 'banjir': np.int64(1), 'datang': np.int64(1), 'korban': np.int64(0)}
```

Catatan: tampilan nilai dapat berbeda antarversi NumPy. Secara konseptual, nilai-nilai tersebut adalah bilangan bulat.

### 6.5 Parameter Penting CountVectorizer

| Parameter | Fungsi |
|---|---|
| `lowercase` | Mengubah teks menjadi huruf kecil; default `True`. |
| `stop_words` | Menentukan stopword yang akan diabaikan, jika sesuai kebutuhan. |
| `ngram_range` | Menentukan rentang n-gram, misalnya unigram atau bigram. |
| `min_df` | Mengabaikan token yang muncul pada terlalu sedikit dokumen. |
| `max_df` | Mengabaikan token yang muncul pada terlalu banyak dokumen. |
| `max_features` | Membatasi jumlah fitur yang dipilih. |
| `binary` | Jika `True`, nilai fitur menunjukkan keberadaan token, bukan jumlah kemunculan. |

Contoh penggunaan bigram:

```python
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "butuh bantuan segera",
    "bantuan segera datang"
]

vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

Output:

```text
['bantuan' 'bantuan segera' 'butuh' 'butuh bantuan'
 'datang' 'segera' 'segera datang']
[[0 0 1 1 0 1 0]
 [1 1 0 0 1 1 1]]
```

Bigram adalah pasangan dua token yang berurutan, seperti `butuh bantuan` dan `segera datang`. Perhatikan bahwa n-gram dapat membantu mempertahankan sebagian informasi frasa, tetapi tidak sepenuhnya memahami makna kalimat.

## 7. Term Frequency (TF)

### 7.1 Konsep

**Term Frequency (TF)** mengukur seberapa sering suatu term muncul dalam sebuah document. Ada beberapa definisi TF. Bentuk sederhana yang digunakan di sini adalah frekuensi term dibagi jumlah seluruh token dalam document.

\[
TF(t,d)=\frac{\text{jumlah kemunculan }t\text{ di }d}
{\text{jumlah seluruh token di }d}
\]

Misalnya document:

```text
"banjir banjir butuh bantuan"
```

Jumlah token = 4.

- TF(`banjir`) = 2/4 = 0.5
- TF(`butuh`) = 1/4 = 0.25
- TF(`bantuan`) = 1/4 = 0.25

### 7.2 Implementasi Manual

```python
from collections import Counter

document = "banjir banjir butuh bantuan".split()
counts = Counter(document)
total_tokens = len(document)

for term, count in counts.items():
    tf = count / total_tokens
    print(f"TF({term}) = {count}/{total_tokens} = {tf:.2f}")
```

Output:

```text
TF(banjir) = 2/4 = 0.50
TF(butuh) = 1/4 = 0.25
TF(bantuan) = 1/4 = 0.25
```

TF memberi bobot berdasarkan frekuensi dalam satu document, tetapi belum mempertimbangkan apakah kata tersebut umum di seluruh corpus.

## 8. Inverse Document Frequency (IDF)

### 8.1 Konsep

**Inverse Document Frequency (IDF)** mengukur seberapa informatif suatu term berdasarkan seberapa banyak document yang memuatnya. Kata yang muncul di hampir semua document cenderung memperoleh IDF lebih rendah.

Salah satu rumus IDF yang umum adalah:

\[
IDF(t)=\log\left(\frac{N}{DF(t)}\right)
\]

Keterangan:

- \(N\): jumlah seluruh document.
- \(DF(t)\): jumlah document yang mengandung term \(t\).
- \(\log\): logaritma natural atau basis lain, bergantung implementasi.

Pada rumus dasar tersebut, jika suatu term muncul di semua document, IDF-nya adalah 0. Dalam praktik, library sering memakai varian dengan smoothing agar perhitungan lebih stabil.

### 8.2 Contoh Manual

Corpus:

```text
Document 1: "banjir butuh bantuan"
Document 2: "banjir merendam rumah"
Document 3: "cuaca cerah"
```

Jumlah document \(N=3\).

- `banjir` muncul pada 2 document.
- `bantuan` muncul pada 1 document.
- `cuaca` muncul pada 1 document.

Dengan rumus dasar:

\[
IDF(\text{banjir})=\log(3/2)\approx 0.405
\]

\[
IDF(\text{bantuan})=\log(3/1)\approx 1.099
\]

\[
IDF(\text{cuaca})=\log(3/1)\approx 1.099
\]

Jadi `bantuan` dan `cuaca` memiliki IDF lebih tinggi daripada `banjir` dalam corpus kecil ini karena keduanya muncul di lebih sedikit document.

### 8.3 Implementasi Manual

```python
import math

documents = [
    "banjir butuh bantuan",
    "banjir merendam rumah",
    "cuaca cerah"
]

terms = ["banjir", "bantuan", "cuaca"]
N = len(documents)

for term in terms:
    df = sum(term in document.split() for document in documents)
    idf = math.log(N / df)
    print(f"IDF({term}) = log({N}/{df}) = {idf:.3f}")
```

Output:

```text
IDF(banjir) = log(3/2) = 0.405
IDF(bantuan) = log(3/1) = 1.099
IDF(cuaca) = log(3/1) = 1.099
```

Perhitungan ini menggunakan rumus IDF dasar tanpa smoothing. Nilai IDF dari `TfidfVectorizer` bisa berbeda karena library menggunakan rumus yang berbeda.

## 9. TF-IDF

### 9.1 Konsep

**TF-IDF** menggabungkan Term Frequency dan Inverse Document Frequency. Tujuannya memberi bobot yang mempertimbangkan frekuensi term pada sebuah document sekaligus kelangkaannya pada corpus.

\[
TF\text{-}IDF(t,d)=TF(t,d)\times IDF(t)
\]

Intuisi:

- Term sering muncul dalam sebuah document dapat memperoleh TF tinggi.
- Term yang tersebar di hampir semua document cenderung memperoleh IDF rendah.
- Term yang cukup sering muncul pada document tertentu tetapi jarang pada document lain dapat memperoleh bobot relatif tinggi.

TF-IDF **bukan** ukuran pemahaman semantik. Kata-kata yang bermakna mirip belum tentu memiliki vektor yang mirip.

### 9.2 Contoh Perhitungan Sederhana

Misalnya:

- TF(`bantuan`, Document 1) = 1/3
- IDF(`bantuan`) = \(\log(3/1)\approx 1.099\)

Maka:

\[
TF\text{-}IDF(\text{bantuan},d_1)
=\frac{1}{3}\times 1.099
\approx 0.366
\]

Ini hanya ilustrasi dengan TF yang dinormalisasi berdasarkan panjang document dan IDF dasar tanpa smoothing. Nilai dari scikit-learn bisa berbeda.

### 9.3 Implementasi dengan TfidfVectorizer

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "banjir butuh bantuan",
    "banjir merendam rumah",
    "cuaca cerah"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

print("Vocabulary:")
print(vectorizer.get_feature_names_out())

print("\nMatriks TF-IDF:")
print(X.toarray().round(3))
```

Output:

```text
Vocabulary:
['bantuan' 'banjir' 'butuh' 'cerah' 'merendam' 'rumah' 'cuaca']

Matriks TF-IDF:
[[0.629 0.449 0.629 0.    0.    0.    0.   ]
 [0.    0.449 0.    0.    0.631 0.631 0.   ]
 [0.    0.    0.    0.707 0.    0.    0.707]]
```

Angka dapat sedikit berbeda akibat versi library atau pengaturan parameter. Pada konfigurasi default scikit-learn, IDF menggunakan smoothing dan vektor hasil akhirnya dinormalisasi. Karena itu, nilai tidak harus sama dengan perhitungan manual pada bagian sebelumnya.

### 9.4 Mengapa Nilai TF-IDF Bisa Berbeda?

Perbedaan dapat terjadi karena:

1. Rumus IDF yang digunakan.
2. Apakah panjang document dinormalisasi.
3. Pengaturan smoothing.
4. Pengaturan normalisasi vektor, seperti `norm='l2'`.
5. Tokenisasi dan preprocessing yang berbeda.

Jadi, ketika melakukan penelitian, dokumentasikan parameter representasi yang digunakan.

## 10. Perbedaan BoW, CountVectorizer, dan TF-IDF

| Metode | Representasi | Memperhitungkan frekuensi | Mempertimbangkan kelangkaan antar-document |
|---|---|---|---|
| One-Hot | Vektor identitas token | Tidak untuk satu token | Tidak |
| BoW manual | Hitungan token per document | Ya | Tidak |
| CountVectorizer | Matriks hitungan token | Ya | Tidak |
| TF-IDF | Bobot token | Ya, melalui TF | Ya, melalui IDF |

`CountVectorizer` adalah implementasi praktis untuk membangun representasi berbasis hitungan. TF-IDF mengubah hitungan tersebut menjadi bobot dengan memperhitungkan distribusi term pada corpus.

## 11. Sparse Matrix

Vocabulary yang besar menghasilkan banyak nilai nol. Matriks yang sebagian besar berisi nol disebut **sparse matrix**.

Contoh:

```text
[1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
```

Karena sebagian besar nilainya nol, scikit-learn umumnya menyimpan hasil `CountVectorizer` dan `TfidfVectorizer` dalam format sparse agar penggunaan memori lebih efisien.

Untuk melihat bentuk matriks:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "tolong bantu kami",
    "banjir butuh bantuan",
    "cuaca cerah hari ini"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

print("Shape:", X.shape)
print("Jumlah nilai non-zero:", X.nnz)
print("Jumlah seluruh elemen:", X.shape[0] * X.shape[1])
```

Output:

```text
Shape: (3, 8)
Jumlah nilai non-zero: 8
Jumlah seluruh elemen: 24
```

Artinya terdapat 3 document dan 8 fitur, atau 24 posisi dalam matriks padat. Hanya 8 posisi yang memiliki nilai non-zero. Jumlah persis fitur dan non-zero mengikuti corpus dan tokenisasi pada contoh.

Gunakan `X.toarray()` untuk melihat matriks kecil saat belajar. Untuk dataset besar, hindari mengubah seluruh sparse matrix menjadi array padat karena bisa membutuhkan banyak memori.

## 12. Preprocessing Sebelum Text Representation

Preprocessing memengaruhi vocabulary dan hasil representasi. Misalnya:

```text
"TOLONG bantu kami!!!"
"tolong bantu kami"
```

Tanpa normalisasi, variasi kapitalisasi atau tanda baca dapat menghasilkan fitur berbeda, bergantung metode tokenisasi.

Contoh sederhana:

```python
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "TOLONG bantu kami!!!",
    "tolong bantu kami"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

Output:

```text
['bantu' 'kami' 'tolong']
[[1 1 1]
 [1 1 1]]
```

Dalam contoh ini, `CountVectorizer` secara default mengubah teks menjadi huruf kecil dan tanda baca tidak menjadi fitur kata tersendiri, sehingga kedua document menghasilkan hitungan yang sama.

### Catatan penting untuk IndoBERT

Preprocessing yang cocok untuk TF-IDF belum tentu cocok untuk IndoBERT.

Untuk TF-IDF, normalisasi, penghapusan URL, dan penanganan bahasa informal bisa membantu. Untuk IndoBERT, hindari menghapus kata atau tanda yang mungkin mengandung informasi penting secara otomatis. Misalnya, kata negasi seperti `tidak`, `bukan`, dan `belum` dapat mengubah makna kalimat.

Jangan menerapkan stemming, stopword removal, atau penghapusan emoji secara membabi buta. Tentukan langkah berdasarkan karakteristik dataset, dokumentasi tokenizer, dan eksperimen yang terukur.

## 13. Menggunakan Text Representation untuk Klasifikasi

Representasi teks dapat dipasangkan dengan model Machine Learning klasik.

Alur:

```text
Teks
  ↓
TfidfVectorizer
  ↓
Matriks fitur
  ↓
Logistic Regression / SVM
  ↓
Prediksi kelas
```

Contoh kode lengkap dengan dataset mainan:

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

texts = [
    "tolong bantu kami banjir",
    "butuh bantuan makanan",
    "rumah terendam banjir",
    "kami terjebak banjir",
    "hari ini cuaca cerah",
    "saya sedang belajar",
    "mari makan bersama",
    "informasi acara besok"
]

labels = [
    "emergency",
    "emergency",
    "emergency",
    "emergency",
    "non_emergency",
    "non_emergency",
    "non_emergency",
    "non_emergency"
]

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Prediksi:", predictions)
print("Label aktual:", list(y_test))
print(classification_report(y_test, predictions, zero_division=0))
```

Output:

```text
Prediksi: ['non_emergency' 'emergency']
Label aktual: ['non_emergency', 'emergency']

              precision    recall  f1-score   support

    emergency       1.00      1.00      1.00         1
non_emergency       1.00      1.00      1.00         1

    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2
```

**Penting:** output metrik di atas hanya ilustrasi untuk dataset sangat kecil. Skor sempurna pada dua contoh uji tidak membuktikan model benar-benar baik. Penelitian memerlukan dataset representatif yang lebih besar, pembagian data yang tepat, dan evaluasi yang memadai.

Pipeline memastikan `TfidfVectorizer` dipelajari dari data training ketika `fit()` dipanggil. Jangan melakukan `fit_transform()` pada seluruh dataset sebelum train/test split karena hal itu dapat menyebabkan kebocoran informasi dari data test.

## 14. Hubungan Text Representation dengan Penelitian Anda

Pada penelitian deteksi tweet permintaan bantuan darurat, TF-IDF dapat digunakan sebagai **baseline** untuk dibandingkan dengan IndoBERT.

Contoh rancangan eksperimen:

```text
Dataset tweet berlabel
       ↓
Train / Validation / Test split
       ↓
 ┌───────────────────────┐
 │ Baseline klasik       │
 │ TF-IDF + Logistic Reg.│
 │ TF-IDF + SVM          │
 └───────────────────────┘
       ↓
Evaluasi
       ↓
Bandingkan dengan IndoBERT
```

Hal yang perlu dijaga:

1. Gunakan pembagian data yang sama untuk perbandingan model, bila sesuai dengan rancangan eksperimen.
2. Fit TF-IDF hanya pada data training.
3. Pilih parameter berdasarkan validation set atau cross-validation, bukan test set.
4. Laporkan precision, recall, F1-score, dan confusion matrix, bukan accuracy saja.
5. Perhatikan recall kelas `emergency`, karena false negative dapat berarti permintaan bantuan tidak terdeteksi.
6. Jika tweet berasal dari kejadian, pengguna, atau waktu yang saling terkait, pertimbangkan potensi kemiripan antar-split dan pilih strategi pemisahan yang mencegah evaluasi terlalu optimistis.
7. Tetapkan aturan anotasi dan kualitas label sebelum menarik kesimpulan dari hasil model.

TF-IDF tidak perlu mengalahkan IndoBERT untuk tetap berguna. Baseline menunjukkan seberapa jauh metode yang lebih kompleks meningkatkan performa pada data dan protokol evaluasi yang sama.

## 15. Keterbatasan TF-IDF

TF-IDF kuat sebagai baseline, tetapi mempunyai beberapa keterbatasan:

- Tidak memahami konteks kata.
- Tidak secara langsung memahami sinonim.
- Representasi kata biasanya sama meskipun konteksnya berbeda.
- Urutan kata umumnya tidak dipertahankan, kecuali sebagian informasi ditambahkan melalui n-gram.
- Kata baru atau variasi ejaan dapat menjadi masalah jika tidak tertangani oleh vocabulary.

Contoh:

```text
"rumah saya terendam banjir"
"rumah saya tidak terendam banjir"
```

Kedua kalimat berbagi banyak kata. TF-IDF tidak memahami negasi secara semantik; model klasifikasi masih bisa mempelajari pola kata jika tersedia data yang memadai, tetapi tidak memiliki pemahaman konteks seperti model Transformer.

Keterbatasan ini menjadi salah satu alasan mempelajari Word Embedding dan Transformer pada tahap berikutnya.

## 16. Latihan Mandiri

### Latihan 1 — Vocabulary

Diberikan:

```python
documents = [
    "bantu korban banjir",
    "korban butuh makanan",
    "banjir datang lagi"
]
```

Tugas:

1. Tentukan vocabulary unik.
2. Hitung jumlah document.
3. Hitung document frequency untuk kata `korban` dan `banjir`.

### Latihan 2 — BoW

Gunakan vocabulary berikut:

```python
vocabulary = ["bantu", "banjir", "korban", "makanan"]
```

Buat fungsi yang mengubah document `"bantu korban banjir"` menjadi vektor BoW.

**Output yang diharapkan:**

```text
[1, 1, 1, 0]
```

### Latihan 3 — CountVectorizer

Gunakan corpus:

```python
documents = [
    "tolong bantu korban",
    "korban butuh bantuan",
    "banjir butuh bantuan"
]
```

Tugas:

1. Buat matriks dengan `CountVectorizer`.
2. Tampilkan nama fitur.
3. Tampilkan matriks hitungan.
4. Jelaskan arti satu baris dan satu kolom.

### Latihan 4 — TF-IDF

Gunakan corpus yang sama dengan Latihan 3.

Tugas:

1. Buat `TfidfVectorizer`.
2. Cetak nama fitur.
3. Cetak matriks TF-IDF dengan tiga angka desimal.
4. Jelaskan mengapa bobot satu kata dapat berbeda antar-document.

### Latihan 5 — Eksperimen N-gram

Bandingkan:

```python
CountVectorizer(ngram_range=(1, 1))
```

dengan:

```python
CountVectorizer(ngram_range=(1, 2))
```

Gunakan teks:

```text
"butuh bantuan segera"
"bantuan segera datang"
```

Catat perbedaan jumlah dan nama fitur.

## 17. Checklist Penguasaan

Tandai setiap poin setelah Anda dapat menjelaskan atau mempraktikkannya tanpa menyalin contoh secara mentah.

- [ ] Saya memahami alasan teks perlu diubah menjadi angka.
- [ ] Saya dapat menjelaskan document, corpus, token, vocabulary, dan feature vector.
- [ ] Saya memahami One-Hot Encoding.
- [ ] Saya dapat membangun BoW secara manual.
- [ ] Saya dapat menggunakan `CountVectorizer`.
- [ ] Saya memahami TF dan document frequency.
- [ ] Saya dapat menjelaskan IDF dan TF-IDF.
- [ ] Saya dapat menggunakan `TfidfVectorizer`.
- [ ] Saya memahami sparse matrix dan vocabulary.
- [ ] Saya dapat memasukkan TF-IDF ke dalam pipeline klasifikasi.
- [ ] Saya memahami mengapa preprocessing harus dipilih dengan hati-hati.
- [ ] Saya memahami bahwa TF-IDF adalah baseline, bukan representasi yang memahami konteks secara penuh.

## 18. Ringkasan

Perjalanan materi ini:

```text
Text
 ↓
Token / Vocabulary
 ↓
One-Hot Encoding
 ↓
Bag of Words
 ↓
CountVectorizer
 ↓
TF + IDF
 ↓
TF-IDF
 ↓
Feature Matrix
 ↓
Machine Learning Classifier
```

Konsep inti:

- **One-Hot Encoding** menandai identitas token menggunakan vektor.
- **Bag of Words** merepresentasikan document dengan hitungan kata dan mengabaikan urutan kata.
- **CountVectorizer** membangun matriks hitungan token secara otomatis.
- **TF** mengukur frekuensi term dalam document.
- **IDF** mengukur kelangkaan term antar-document.
- **TF-IDF** menggabungkan frekuensi dan kelangkaan menjadi bobot fitur.
- **Sparse matrix** menyimpan representasi yang banyak berisi nilai nol secara lebih efisien.
- **TF-IDF + Logistic Regression/SVM** merupakan baseline yang layak untuk dibandingkan dengan IndoBERT dalam penelitian klasifikasi tweet.
