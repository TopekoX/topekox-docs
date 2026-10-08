# Fundamental NLP untuk Penelitian Deteksi Tweet Permintaan Bantuan Darurat

## Tujuan Pembelajaran

Setelah menyelesaikan materi ini, Anda diharapkan mampu:

- Menjelaskan apa itu Natural Language Processing (NLP).
- Memahami hubungan antara teks, dokumen, kalimat, token, dan vocabulary.
- Memahami konsep corpus dan dataset NLP.
- Memahami mengapa teks perlu diubah menjadi representasi yang dapat diproses komputer.
- Memahami gambaran umum pipeline NLP.
- Menjelaskan hubungan Fundamental NLP dengan penelitian deteksi tweet permintaan bantuan darurat.
- Memahami konsep dasar yang akan digunakan pada tahap Text Preprocessing, TF-IDF, Machine Learning, Transformer, dan IndoBERT.

Materi ini sengaja tidak membahas seluruh cabang NLP secara mendalam. Fokusnya adalah konsep yang benar-benar diperlukan sebagai fondasi penelitian.

## 1. Apa Itu Natural Language Processing

Natural Language Processing atau NLP adalah bidang dalam Artificial Intelligence (AI) yang mempelajari bagaimana komputer dapat memproses, memahami, menganalisis, dan menghasilkan bahasa manusia.

Bahasa manusia memiliki karakteristik yang kompleks. Satu kata dapat memiliki makna berbeda tergantung konteks, manusia dapat menggunakan singkatan, typo, bahasa informal, emoji, dan susunan kalimat yang beragam.

Contoh teks:

```text
Tolong bantu kami, rumah sudah terendam banjir.
```

Manusia dapat memahami bahwa kalimat tersebut kemungkinan merupakan permintaan bantuan.

Komputer tidak memahami kalimat tersebut dengan cara yang sama seperti manusia. Teks perlu melalui beberapa proses agar dapat direpresentasikan dalam bentuk yang dapat dipelajari oleh model Machine Learning atau Deep Learning.

Secara sederhana:

```text
Bahasa Manusia
      ↓
      NLP
      ↓
Representasi yang dapat diproses komputer
      ↓
Machine Learning / Deep Learning
      ↓
Prediksi / Analisis
```

## 2. NLP dalam Penelitian Ini

Penelitian yang akan dikembangkan adalah:

```text
Deteksi Tweet Permintaan Bantuan Darurat
pada Saat Bencana Menggunakan IndoBERT
```

Contoh tweet:

```text
Tolong bantu kami, rumah sudah terendam banjir.
```

Tujuan model adalah menentukan apakah tweet tersebut merupakan permintaan bantuan darurat.

Contoh sederhana:

```text
Input:
"Tolong bantu kami, rumah sudah terendam banjir."

Output:
Emergency
```

Sedangkan:

```text
Input:
"Hujan deras terjadi sejak pagi di wilayah Palu."

Output:
Non-Emergency
```

Secara umum, masalah ini termasuk:

```text
NLP
 ↓
Text Classification
 ↓
Binary Classification
```

Jika menggunakan dua kelas:

```text
0 = Non-Emergency
1 = Emergency
```

## 3. Text

Text atau teks adalah data berupa bahasa yang digunakan sebagai input dalam sistem NLP.

Contoh:

```text
Tolong bantu kami.
```

```text
Banjir mulai memasuki rumah warga.
```

```text
Jalan utama terputus akibat longsor.
```

Dalam penelitian media sosial, teks dapat berasal dari:

- Tweet
- Postingan media sosial
- Komentar
- Berita
- Laporan masyarakat
- Pesan atau dokumen lainnya

Dalam penelitian ini, sumber utama yang digunakan adalah tweet atau teks pendek dari media sosial.

## 4. Document

Dalam NLP, sebuah document adalah satu unit teks yang diperlakukan sebagai satu data.

Document tidak harus berupa dokumen panjang seperti file PDF atau Microsoft Word.

Satu tweet juga dapat dianggap sebagai satu document.

Contoh:

```text
"Tolong bantu kami, rumah sudah terendam banjir."
```

Dapat dianggap sebagai:

```text
1 Document
```

Jika terdapat 5.000 tweet:

```text
Tweet 1 → Document 1
Tweet 2 → Document 2
Tweet 3 → Document 3
...
Tweet 5000 → Document 5000
```

Dengan demikian, dataset klasifikasi teks dapat dipandang sebagai kumpulan document.

## 5. Sentence

Sentence atau kalimat adalah rangkaian kata yang membentuk suatu struktur bahasa.

Contoh:

```text
"Tolong bantu kami."
```

merupakan satu kalimat.

Sebuah document dapat terdiri dari satu atau beberapa kalimat.

Contoh:

```text
"Hujan turun sejak pagi. Air mulai masuk ke rumah warga."
```

Document tersebut terdiri dari dua kalimat.

Dalam tweet, sebagian besar data relatif pendek, tetapi sebuah tweet tetap dapat berisi lebih dari satu kalimat.

## 6. Token

Token adalah unit teks yang dihasilkan ketika teks dipecah menjadi bagian-bagian yang lebih kecil.

Contoh:

```text
Tolong bantu kami
```

dapat dipecah menjadi:

```text
["Tolong", "bantu", "kami"]
```

Masing-masing bagian tersebut disebut token.

Secara sederhana:

```text
Text
 ↓
Tokenization
 ↓
Tokens
```

Contoh:

```text
"Tolong bantu kami karena banjir"
```

menjadi:

```text
[
    "Tolong",
    "bantu",
    "kami",
    "karena",
    "banjir"
]
```

## 7. Mengapa Token Penting

Token merupakan konsep yang sangat penting dalam NLP karena sebagian besar algoritma NLP bekerja dengan unit-unit teks yang telah dipisahkan.

Misalnya:

```text
"Tolong bantu kami"
```

menjadi:

```text
["Tolong", "bantu", "kami"]
```

Setelah itu token dapat diproses lebih lanjut.

Contohnya:

```text
Text
 ↓
Tokenization
 ↓
["tolong", "bantu", "kami"]
 ↓
Numerical Representation
 ↓
Model
```

Konsep token juga sangat penting ketika nanti menggunakan Transformer dan IndoBERT.

Namun, perlu dipahami bahwa tokenisasi pada model modern seperti BERT tidak selalu sama dengan tokenisasi berdasarkan kata.

BERT menggunakan pendekatan subword tokenization.

## 8. Word Tokenization

Word tokenization memecah teks berdasarkan kata.

Contoh:

```text
"Saya membutuhkan bantuan"
```

menjadi:

```text
["Saya", "membutuhkan", "bantuan"]
```

Pendekatan ini cukup mudah dipahami dan berguna untuk mempelajari dasar NLP.

Dalam Python, salah satu pendekatan paling sederhana adalah:

```python
text = "Saya membutuhkan bantuan"

tokens = text.split()

print(tokens)
```

Output:

```text
['Saya', 'membutuhkan', 'bantuan']
```

Namun, `split()` masih sangat sederhana dan belum menangani berbagai karakter khusus, punctuation, emoji, URL, atau struktur bahasa secara baik.

Karena itu, pada tahap berikutnya kita akan mempelajari tokenizer yang lebih sesuai untuk NLP.

## 9. Subword Tokenization

Model Transformer seperti BERT menggunakan pendekatan subword tokenization.

Konsepnya adalah sebuah kata dapat dipecah menjadi bagian-bagian yang lebih kecil apabila diperlukan.

Contoh ilustrasi:

```text
kata
 ↓
subword
```

Misalnya sebuah kata yang tidak terdapat sebagai token utuh dapat direpresentasikan menggunakan beberapa subword.

Konsep ini membantu model menangani:

- Kata yang jarang muncul
- Kata baru
- Kata turunan
- Variasi kata
- Out-of-vocabulary words

Dalam IndoBERT, proses tokenisasi akan dilakukan oleh tokenizer yang sesuai dengan model.

Contoh penggunaan:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "indobenchmark/indobert-base-p1"
)
```

Kemudian:

```python
text = "Tolong bantu kami karena banjir."

tokens = tokenizer.tokenize(text)

print(tokens)
```

Hasil tokenisasi bergantung pada tokenizer dan vocabulary model.

Hal penting yang perlu dipahami sekarang adalah:

> Token pada NLP klasik dan token pada Transformer tidak selalu berarti hal yang sama.

## 10. Vocabulary

Vocabulary adalah kumpulan token yang dikenal oleh suatu sistem atau model.

Misalnya kita memiliki dataset:

```text
Tolong bantu kami
Banjir melanda rumah
Kami membutuhkan bantuan
```

Vocabulary sederhananya dapat berupa:

```text
[
    "tolong",
    "bantu",
    "kami",
    "banjir",
    "melanda",
    "rumah",
    "membutuhkan",
    "bantuan"
]
```

Vocabulary dapat dianggap sebagai daftar unit bahasa yang diketahui oleh sistem.

Dalam model Transformer seperti IndoBERT, vocabulary sudah dibangun sebagai bagian dari tokenizer dan model yang digunakan.

## 11. Corpus

Corpus adalah kumpulan teks yang digunakan untuk analisis atau pengembangan sistem NLP.

Contoh corpus sederhana:

```text
Document 1:
"Tolong bantu kami."

Document 2:
"Rumah warga terendam banjir."

Document 3:
"Jalan utama terputus."

Document 4:
"Hujan deras terjadi sejak pagi."
```

Keempat document tersebut dapat menjadi sebuah corpus.

Secara sederhana:

```text
Corpus
 ├── Document 1
 ├── Document 2
 ├── Document 3
 └── Document 4
```

Dalam penelitian Anda, kumpulan tweet yang telah dikumpulkan dapat diperlakukan sebagai corpus penelitian.

## 12. Dataset NLP

Dataset adalah kumpulan data yang digunakan untuk proses Machine Learning.

Pada penelitian klasifikasi teks, dataset biasanya memiliki minimal dua komponen:

```text
text
label
```

Contoh:

| text | label |
|---|---:|
| Tolong bantu kami, rumah terendam banjir | 1 |
| Jalan utama terputus akibat longsor | 1 |
| Hujan deras terjadi sejak pagi | 0 |
| Kondisi cuaca hari ini cukup cerah | 0 |

Keterangan:

```text
1 = Emergency
0 = Non-Emergency
```

Dataset inilah yang nantinya digunakan untuk melatih model.

## 13. Feature dan Label

Dalam Machine Learning, kita sering mengenal:

```text
X = Features
y = Target / Label
```

Dalam klasifikasi teks:

```text
X = text
y = label
```

Contoh:

```text
X:
"Tolong bantu kami, rumah terendam banjir"

y:
1
```

Namun teks mentah belum dapat langsung digunakan oleh sebagian besar algoritma Machine Learning klasik.

Karena itu:

```text
Text
 ↓
Text Representation
 ↓
Numerical Features
 ↓
Machine Learning Model
```

Contoh:

```text
Tweet
 ↓
TF-IDF
 ↓
Feature Matrix
 ↓
SVM
 ↓
Prediction
```

Untuk Transformer:

```text
Tweet
 ↓
Tokenizer
 ↓
Input IDs + Attention Mask
 ↓
IndoBERT
 ↓
Prediction
```

## 14. Text Classification

Text Classification adalah proses mengelompokkan teks ke dalam kategori tertentu.

Contoh:

```text
Email
 ↓
Spam / Not Spam
```

Contoh lainnya:

```text
Review
 ↓
Positive / Negative
```

Untuk penelitian Anda:

```text
Tweet
 ↓
Emergency / Non-Emergency
```

Dengan demikian:

```text
Deteksi Tweet Permintaan Bantuan Darurat
=
Text Classification
```

## 15. Binary Classification

Jika hanya terdapat dua kelas, masalah tersebut disebut binary classification.

Contoh:

```text
0 = Non-Emergency
1 = Emergency
```

Model menerima:

```text
"Tolong bantu kami, rumah sudah terendam banjir."
```

dan menghasilkan prediksi:

```text
Emergency
```

Jika model memberikan probabilitas:

```text
Emergency     : 0.94
Non-Emergency : 0.06
```

maka prediksi akhirnya adalah:

```text
Emergency
```

## 16. Multi-Class Classification

Jika kategori lebih dari dua, masalah dapat menjadi multi-class classification.

Contoh:

```text
0 = Informational
1 = Emergency
2 = Infrastructure Damage
3 = Evacuation
```

Contoh:

```text
"Jembatan utama putus akibat banjir."
```

dapat dikategorikan sebagai:

```text
Infrastructure Damage
```

Untuk tahap awal penelitian, binary classification lebih sederhana dan lebih mudah dikontrol.

Setelah pipeline dasar berhasil, klasifikasi dapat dikembangkan menjadi beberapa kategori apabila dataset dan tujuan penelitian mendukungnya.

## 17. NLP Pipeline

NLP pipeline adalah rangkaian proses yang dilakukan terhadap teks dari input hingga menghasilkan output.

Untuk Machine Learning klasik:

```text
Raw Text
    ↓
Cleaning
    ↓
Tokenization
    ↓
Normalization
    ↓
Feature Extraction
    ↓
Machine Learning
    ↓
Prediction
```

Contoh:

```text
"TOLONG!!! rumah kami kebanjiran 😭"
                ↓
             Cleaning
                ↓
"tolong rumah kami kebanjiran"
                ↓
             Tokenization
                ↓
["tolong", "rumah", "kami", "kebanjiran"]
                ↓
              TF-IDF
                ↓
               SVM
                ↓
           "Emergency"
```

Untuk IndoBERT, pipeline akan berbeda:

```text
Raw Text
    ↓
Minimal / Task-Specific Preprocessing
    ↓
IndoBERT Tokenizer
    ↓
Input IDs
    ↓
Attention Mask
    ↓
IndoBERT
    ↓
Classification Head
    ↓
Prediction
```

Perbedaan ini sangat penting.

Jangan menganggap preprocessing untuk TF-IDF harus diterapkan persis sama kepada IndoBERT.

## 18. NLP Klasik vs Transformer

Kita dapat membandingkan dua pendekatan.

### NLP Klasik

```text
Text
 ↓
Cleaning
 ↓
Tokenization
 ↓
Stopword / Stemming
 ↓
TF-IDF
 ↓
SVM
 ↓
Prediction
```

Contoh:

```text
TF-IDF + SVM
```

### Transformer

```text
Text
 ↓
Tokenizer
 ↓
Input IDs
 ↓
Attention Mask
 ↓
Transformer
 ↓
Classification Head
 ↓
Prediction
```

Contoh:

```text
IndoBERT
```

Perbedaan ini akan menjadi dasar eksperimen penelitian Anda.

## 19. Mengapa Tidak Langsung Menggunakan IndoBERT

Pertanyaan penting:

> Jika IndoBERT lebih modern, mengapa perlu belajar TF-IDF dan Machine Learning klasik?

Karena dalam penelitian ilmiah, model utama sebaiknya memiliki baseline atau pembanding.

Misalnya:

```text
Baseline 1:
TF-IDF + Logistic Regression

Baseline 2:
TF-IDF + SVM

Model utama:
IndoBERT
```

Kemudian hasil dibandingkan:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | ... | ... | ... | ... |
| TF-IDF + SVM | ... | ... | ... | ... |
| IndoBERT | ... | ... | ... | ... |

Dengan demikian, penelitian tidak hanya mengatakan:

> "IndoBERT dapat digunakan untuk klasifikasi tweet."

Tetapi dapat menjawab:

> "Apakah IndoBERT memberikan performa yang lebih baik dibandingkan metode Machine Learning berbasis TF-IDF?"

## 20. NLP dan Konteks Bahasa Indonesia

Bahasa Indonesia memiliki karakteristik yang perlu diperhatikan dalam NLP.

Contoh:

```text
bantu
membantu
membantukan
dibantu
membutuhkan bantuan
```

Ada hubungan morfologis dan konteks yang tidak selalu dapat ditangkap hanya dengan melihat kata secara terpisah.

Selain itu, media sosial memiliki karakteristik tambahan:

```text
tolonggg
bgt
gk
ga
nggak
RT
@username
#banjir
😭
!!!
```

Karena penelitian menggunakan tweet, masalah seperti ini akan muncul dalam dataset.

Karena itu, tahap berikutnya setelah Fundamental NLP adalah:

```text
Text Preprocessing
```

## 21. Konteks Lebih Penting daripada Kata Saja

Salah satu alasan NLP modern berkembang menuju model contextual language representation adalah karena makna kata dapat bergantung pada konteks.

Contoh:

```text
"Air sungai naik."
```

dan:

```text
"Air minum naik harganya."
```

Kata yang sama dapat muncul dalam konteks berbeda.

Pada NLP klasik seperti Bag of Words dan TF-IDF, representasi kata cenderung tidak memahami konteks secara mendalam.

Model Transformer seperti BERT menggunakan konteks dua arah untuk membentuk representasi yang lebih kontekstual.

Konsep inilah yang nantinya akan membawa kita menuju:

```text
Word Embedding
      ↓
Contextual Embedding
      ↓
Attention
      ↓
Transformer
      ↓
BERT
      ↓
IndoBERT
```

## 22. Contoh Sederhana Pipeline Penelitian

Bayangkan terdapat tweet:

```text
"Tolong bantu kami!!! Air sudah masuk rumah 😭"
```

Pada tahap penelitian, prosesnya secara konseptual:

```text
Tweet
 ↓
Data Collection
 ↓
Annotation
 ↓
Preprocessing
 ↓
Tokenization
 ↓
Representation
 ↓
Classification
 ↓
Evaluation
```

Untuk baseline:

```text
Tweet
 ↓
Preprocessing
 ↓
TF-IDF
 ↓
SVM
 ↓
Emergency / Non-Emergency
```

Untuk model utama:

```text
Tweet
 ↓
Tokenizer
 ↓
IndoBERT
 ↓
Classification Head
 ↓
Emergency / Non-Emergency
```

## 23. Istilah yang Wajib Dikuasai

Sebelum melanjutkan ke Text Preprocessing, pastikan Anda memahami istilah berikut.

### Text

Data berupa bahasa manusia.

### Document

Satu unit teks yang diproses sebagai satu data.

### Sentence

Kalimat yang terdapat dalam document.

### Token

Unit teks hasil proses tokenization.

### Tokenization

Proses memecah teks menjadi token.

### Vocabulary

Kumpulan token yang diketahui oleh sistem atau model.

### Corpus

Kumpulan document atau teks.

### Dataset

Kumpulan data yang digunakan untuk proses Machine Learning.

### Feature

Informasi yang digunakan model sebagai input.

### Label

Target atau kelas yang ingin diprediksi.

### Text Classification

Proses mengelompokkan teks ke kategori tertentu.

### Binary Classification

Klasifikasi dengan dua kelas.

### NLP Pipeline

Rangkaian proses dari teks mentah sampai menghasilkan output.

## 24. Latihan 1 — Analisis Teks Sederhana

Gunakan teks:

```text
"Tolong bantu kami, rumah sudah terendam banjir."
```

Tentukan:

1. Berapa document?
2. Berapa sentence?
3. Apa saja tokennya?
4. Apa vocabulary-nya?
5. Apa kemungkinan labelnya jika digunakan pada penelitian?

Jawaban yang diharapkan secara konseptual:

```text
Document:
1

Sentence:
1

Tokens:
Tolong
bantu
kami
rumah
sudah
terendam
banjir

Label:
Emergency
```

Perhatikan bahwa vocabulary pada satu document sederhana dapat sama dengan kumpulan token unik document tersebut. Pada dataset sebenarnya, vocabulary dibangun dari kumpulan banyak document dan bergantung pada metode tokenisasi.

## 25. Latihan 2 — Kelompokkan Tweet

Tentukan apakah tweet berikut termasuk Emergency atau Non-Emergency.

```text
1. "Tolong bantu warga di sini, air sudah masuk rumah."

2. "Hujan deras turun sejak pagi."

3. "Kami membutuhkan evakuasi segera."

4. "Pemerintah mengimbau warga tetap waspada."

5. "Jembatan desa kami putus, mohon bantuan."
```

Contoh label:

```text
1 → Emergency
2 → Non-Emergency
3 → Emergency
4 → Non-Emergency
5 → Emergency
```

Perhatikan bahwa beberapa kasus dapat bersifat ambigu. Dalam penelitian nyata, aturan anotasi harus dibuat terlebih dahulu agar proses labeling konsisten.

## 26. Latihan 3 — Buat Dataset Sederhana

Buat DataFrame:

```python
import pandas as pd

data = {
    "text": [
        "Tolong bantu kami, rumah terendam banjir",
        "Hujan deras sejak pagi",
        "Kami membutuhkan evakuasi segera",
        "Cuaca hari ini cukup cerah",
        "Jalan desa kami terputus, mohon bantuan"
    ],
    "label": [
        1,
        0,
        1,
        0,
        1
    ]
}

df = pd.DataFrame(data)

print(df)
```

Target pemahaman:

```text
text
 ↓
X

label
 ↓
y
```

## 27. Latihan 4 — Tokenisasi Sederhana dengan Python

Gunakan:

```python
text = "Tolong bantu kami karena rumah terendam banjir"

tokens = text.lower().split()

print(tokens)
```

Coba ubah teks menjadi:

```text
"TOLONG BANTU KAMI!!!"
```

Kemudian amati hasil:

```python
text.lower().split()
```

Apakah hasilnya sudah ideal?

Jawabannya belum.

Masalah seperti punctuation akan menjadi pembahasan pada materi Text Preprocessing.

## 28. Latihan 5 — Buat NLP Pipeline Sederhana

Buat program dengan alur:

```text
Input Text
    ↓
Lowercase
    ↓
Split Token
    ↓
Hitung Jumlah Token
    ↓
Tampilkan Token
```

Contoh:

```python
text = "Tolong bantu kami karena rumah terendam banjir"

text = text.lower()

tokens = text.split()

print("Text:", text)
print("Tokens:", tokens)
print("Jumlah token:", len(tokens))
```

Tujuan latihan bukan membuat sistem NLP lengkap, tetapi memahami bahwa teks melewati beberapa tahap sebelum dianalisis.

## 29. Mini Project — Text Analyzer

Buat program sederhana yang menerima sebuah teks dan menampilkan:

```text
Original Text
Clean Text
Jumlah Karakter
Jumlah Kata
Jumlah Token
Daftar Token
Vocabulary
```

Contoh input:

```text
"Tolong bantu kami, rumah sudah terendam banjir."
```

Contoh output konseptual:

```text
Original Text:
Tolong bantu kami, rumah sudah terendam banjir.

Tokens:
['tolong', 'bantu', 'kami', 'rumah', 'sudah', 'terendam', 'banjir']

Jumlah Token:
7

Vocabulary:
['tolong', 'bantu', 'kami', 'rumah', 'sudah', 'terendam', 'banjir']
```

Mini project ini akan menjadi dasar sebelum masuk ke preprocessing yang lebih serius.

## 30. Checkpoint Fundamental NLP

Sebelum melanjutkan, pastikan Anda dapat menjawab pertanyaan berikut tanpa melihat materi.

### Konsep

1. Apa itu NLP?
2. Apa perbedaan document dan sentence?
3. Apa itu token?
4. Apa itu tokenization?
5. Apa itu vocabulary?
6. Apa itu corpus?
7. Apa perbedaan corpus dan dataset?
8. Apa itu feature?
9. Apa itu label?
10. Apa itu text classification?

### Penelitian

11. Mengapa penelitian Anda termasuk text classification?
12. Apa label yang digunakan?
13. Mengapa tweet harus direpresentasikan dalam bentuk yang dapat diproses model?
14. Apa perbedaan pipeline TF-IDF + SVM dengan IndoBERT?
15. Mengapa baseline penting dalam penelitian?

### Praktik

Anda seharusnya sudah mampu:

```text
Text
 ↓
Tokenization sederhana
 ↓
Tokens
 ↓
Vocabulary
```

dan memahami:

```text
Text
 ↓
Representation
 ↓
Model
 ↓
Prediction
```

## 31. Hubungan Fundamental NLP dengan Roadmap Penelitian

Setelah Fundamental NLP, perjalanan Anda akan menjadi:

```text
Fundamental NLP
      ↓
Text Preprocessing
      ↓
Bag of Words
      ↓
TF-IDF
      ↓
Machine Learning
      ↓
Evaluation
      ↓
Word Embedding
      ↓
PyTorch
      ↓
RNN / LSTM
      ↓
Attention
      ↓
Transformer
      ↓
BERT
      ↓
IndoBERT
      ↓
Fine-Tuning
      ↓
Eksperimen Penelitian
      ↓
FastAPI
      ↓
Dashboard
```

Fundamental NLP adalah fondasi untuk memahami semua tahap tersebut.

## 32. Ringkasan

Hal terpenting yang harus Anda bawa dari materi ini adalah:

```text
Text
 ↓
Document / Sentence
 ↓
Token
 ↓
Vocabulary
 ↓
Representation
 ↓
Model
 ↓
Prediction
```

Untuk penelitian Anda:

```text
Tweet
 ↓
Text Classification
 ↓
Emergency / Non-Emergency
```

Kemudian terdapat dua jalur utama:

```text
Jalur Baseline:

Tweet
 ↓
Preprocessing
 ↓
TF-IDF
 ↓
SVM
 ↓
Prediction
```

dan:

```text
Jalur Model Utama:

Tweet
 ↓
Tokenizer
 ↓
IndoBERT
 ↓
Fine-Tuning
 ↓
Prediction
```

Dengan memahami Fundamental NLP ini, Anda belum menjadi ahli NLP, dan memang belum perlu. Tujuan tahap ini adalah memastikan Anda memahami bahasa dan konsep dasar yang akan digunakan ketika mulai memproses dataset nyata.

Tahap berikutnya adalah **Text Preprocessing**, yang akan menjadi bagian lebih praktis dan sangat relevan dengan karakteristik tweet: URL, mention, hashtag, emoji, slang, typo, repeated characters, stopword, stemming, dan normalisasi bahasa Indonesia.
