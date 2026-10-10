# Text Preprocessing untuk NLP (Natural Language Processing)

## 1. Tujuan Pembelajaran

Setelah menyelesaikan materi ini, Anda diharapkan mampu:

1. Memahami tujuan text preprocessing.
2. Membersihkan data teks dari noise.
3. Melakukan normalisasi teks bahasa Indonesia.
4. Melakukan tokenisasi.
5. Memahami stopword removal.
6. Memahami stemming dan lemmatization.
7. Menangani slang, typo, emoji, URL, mention, dan hashtag.
8. Memahami preprocessing khusus media sosial.
9. Membuat preprocessing pipeline menggunakan Python.
10. Menentukan preprocessing yang tepat untuk **TF-IDF + Machine Learning**.
11. Menentukan preprocessing yang tepat untuk **IndoBERT**.
12. Menghindari preprocessing yang justru merusak informasi penting dalam penelitian.

## 2. Apa Itu Text Preprocessing?

**Text preprocessing** adalah proses mempersiapkan data teks mentah agar lebih sesuai untuk diproses oleh algoritma NLP.

Data teks dari manusia biasanya tidak terstruktur.

Contoh tweet mentah:

```text
TOLONGGG 😭😭😭 @BPBD!!! rumah kami kebanjirannnn...
air sdh setinggi dada!! #BanjirPalu
https://example.com
```

Teks tersebut mengandung:

- huruf kapital,
- karakter berulang,
- emoji,
- mention,
- tanda baca,
- typo,
- singkatan,
- hashtag,
- URL,
- informasi lokasi atau kondisi.

Model machine learning tidak langsung memahami semua bentuk tersebut.

Karena itu kita melakukan preprocessing.

Secara umum:

```text
Raw Text
   ↓
Text Preprocessing
   ↓
Clean / Normalized Text
   ↓
Tokenization
   ↓
Text Representation
   ↓
Machine Learning / Deep Learning
```

## 3. Mengapa Text Preprocessing Penting?

Misalnya dataset Anda memiliki:

```text
Tolong bantu kami
tolong bantu kami
TOLONG BANTU KAMI
tolong bantu kamiiii
```

Secara manusia, kita memahami bahwa semuanya mempunyai makna yang hampir sama.

Namun komputer dapat menganggapnya sebagai bentuk yang berbeda jika tidak ditangani dengan benar.

Preprocessing dapat mengurangi variasi yang tidak diperlukan.

Contohnya:

```text
TOLONG BANTU KAMI!!!
```

menjadi:

```text
tolong bantu kami
```

Tujuan sebenarnya adalah:

> **mengurangi noise sambil mempertahankan informasi yang relevan untuk tugas NLP.**

Ini sangat penting dalam penelitian.

## 4. Prinsip Utama Text Preprocessing

Ada satu prinsip yang harus dipegang:

> **Jangan membersihkan teks secara berlebihan.**

Kesalahan umum pemula adalah berpikir:

```text
Semakin bersih teks → semakin bagus model
```

Tidak selalu benar.

Dalam NLP:

```text
Noise
 ↓
Kurangi
```

tetapi:

```text
Information
 ↓
Pertahankan
```

Contohnya:

```text
"Tolong!!! Rumah kami terendam banjir 😭"
```

Jika kita menghapus:

- `!!!`
- `😭`
- kata `terendam`
- kata `banjir`

maka kita justru menghancurkan informasi penting.

Untuk penelitian deteksi emergency, kata:

```text
tolong
rumah
terendam
banjir
```

sangat mungkin penting.

## 5. Pipeline Text Preprocessing

Pipeline umum:

```text
Raw Text
   ↓
Cleaning
   ↓
Case Folding
   ↓
Normalization
   ↓
Tokenization
   ↓
Stopword Handling
   ↓
Stemming / Lemmatization
   ↓
Final Text
```

Namun pipeline ini **tidak selalu harus digunakan semuanya**.

Untuk penelitian Anda, nanti kita akan membedakan:

```text
TF-IDF Pipeline
```

dan:

```text
IndoBERT Pipeline
```

## 6. Case Folding

### 6.1 Pengertian

Case folding adalah proses mengubah huruf menjadi format yang konsisten, biasanya lowercase.

Contoh:

```text
"TOLONG BANTU KAMI"
```

menjadi:

```text
"tolong bantu kami"
```

### 6.2 Python

```python
text = "TOLONG BANTU KAMI"

text = text.lower()

print(text)
```

Output:

```text
tolong bantu kami
```

### 6.3 Mengapa Diperlukan?

Tanpa case folding:

```text
Banjir
banjir
BANJIR
```

bisa dianggap sebagai token berbeda oleh metode tertentu.

Dengan case folding:

```text
banjir
banjir
banjir
```

Representasi menjadi lebih konsisten.

### 6.4 Apakah Selalu Diperlukan?

Tidak.

Model Transformer seperti BERT menggunakan tokenizer sendiri.

Karena itu jangan otomatis melakukan:

```python
text.lower()
```

sebelum IndoBERT tanpa memahami model dan tokenizer yang digunakan.

Untuk penelitian, prinsip yang lebih baik adalah:

> **ikuti preprocessing yang sesuai dengan model pretrained yang digunakan.**

## 7. Cleaning

Cleaning adalah proses menghapus atau menangani elemen teks yang dianggap sebagai noise.

Contohnya:

- URL
- Mention
- HTML
- Extra whitespace
- Special characters
- Repeated characters

Tetapi setiap elemen harus dipertimbangkan berdasarkan tugas penelitian.

## 8. Menghapus URL

Contoh:

```text
"Tolong bantu korban banjir https://example.com"
```

menjadi:

```text
"Tolong bantu korban banjir"
```

Python:

```python
import re

text = "Tolong bantu korban banjir https://example.com"

text = re.sub(r"http\S+|www\S+", "", text)

print(text)
```

Output:

```text
Tolong bantu korban banjir
```

## 9. Menangani Mention

Tweet dapat mengandung:

```text
@BPBD
@BNPB
@PaluKota
```

Contoh:

```text
"Tolong @BPBD bantu kami"
```

Kita dapat menghapus mention:

```python
text = re.sub(r"@\w+", "", text)
```

Hasil:

```text
"Tolong bantu kami"
```

Namun ada masalah penelitian.

Misalnya:

```text
"Tolong @BPBD bantu kami"
```

`@BPBD` mungkin justru merupakan informasi penting.

Artinya, Anda perlu mempertimbangkan dua pendekatan.

### Pendekatan A — Hapus Mention

```text
"Tolong bantu kami"
```

### Pendekatan B — Ubah Menjadi Token

```text
"Tolong [MENTION] bantu kami"
```

Untuk penelitian, pendekatan B dapat menjadi eksperimen yang menarik.

## 10. Hashtag

Contoh:

```text
#BanjirPalu
#GempaPalu
#TolongKami
```

Jangan langsung menghapus semuanya.

Hashtag dapat mengandung informasi semantik.

Contoh:

```text
#BanjirPalu
```

dapat dinormalisasi menjadi:

```text
banjir palu
```

atau:

```text
[HASHTAG] banjir palu
```

Tergantung eksperimen.

## 11. Menghapus Simbol dan Punctuation

Contoh:

```text
"Tolong!!! Rumah kami terendam!!!"
```

Jika punctuation tidak relevan:

```text
"Tolong Rumah kami terendam"
```

Python:

```python
import string

text = "Tolong!!! Rumah kami terendam!!!"

text = text.translate(
    str.maketrans("", "", string.punctuation)
)

print(text)
```

Output:

```text
Tolong Rumah kami terendam
```

## 12. Hati-Hati Menghapus Punctuation

Jangan berpikir semua tanda baca harus dihapus.

Misalnya:

```text
"Tolong!!!"
```

Jumlah `!` mungkin menunjukkan intensitas atau emotional urgency.

Begitu pula:

```text
"???"
```

atau:

```text
"tolong..."
```

Untuk penelitian klasifikasi emergency, punctuation dapat mengandung sinyal.

Karena itu Anda dapat melakukan eksperimen:

```text
Experiment A:
hapus punctuation

Experiment B:
pertahankan punctuation
```

Kemudian bandingkan F1-score.

Ini lebih ilmiah daripada menentukan preprocessing berdasarkan asumsi.

## 13. Whitespace Normalization

Contoh:

```text
"Tolong     bantu     kami"
```

menjadi:

```text
"Tolong bantu kami"
```

Python:

```python
text = "Tolong     bantu     kami"

text = re.sub(r"\s+", " ", text).strip()

print(text)
```

Output:

```text
Tolong bantu kami
```

Ini umumnya aman dilakukan.

## 14. Menghapus Angka

Contoh:

```text
"Rumah kami di RT 05 terendam 2 meter"
```

Jika semua angka dihapus:

```text
"Rumah kami di RT terendam meter"
```

Informasi hilang.

Untuk penelitian kebencanaan, angka bisa sangat penting:

```text
2 meter
RT 05
10 korban
3 rumah
20 cm
```

Jadi:

> **Jangan otomatis menghapus angka.**

Untuk penelitian Anda, saya justru menyarankan **mempertahankan angka**, setidaknya pada eksperimen awal.

## 15. Emoji

Emoji sangat sering digunakan pada media sosial.

Contoh:

```text
"Tolong kami 😭😭"
```

Emoji:

```text
😭
```

dapat menunjukkan distress atau emotional urgency.

Jadi ada beberapa pilihan.

### Pendekatan 1 — Hapus

```text
"Tolong kami"
```

### Pendekatan 2 — Pertahankan

```text
"Tolong kami 😭😭"
```

### Pendekatan 3 — Konversi

```text
"Tolong kami [CRYING]"
```

Untuk penelitian, Anda bisa membandingkan dampaknya.

## 16. Unicode Normalization

Media sosial dapat mengandung karakter Unicode yang tidak konsisten.

Python menyediakan:

```python
import unicodedata

text = unicodedata.normalize("NFKC", text)
```

Untuk dataset media sosial, Unicode normalization dapat menjadi bagian cleaning pipeline.

## 17. Repeated Characters

Ini sangat penting untuk social media.

Contoh:

```text
"Tolonggggggg"
```

atau:

```text
"BANJIIIIIR"
```

Manusia memahami:

```text
tolong
banjir
```

Kita dapat melakukan normalisasi.

Contoh:

```python
text = re.sub(r"(.)\1{2,}", r"\1\1", text)
```

Misalnya:

```text
tolonggggg
```

dapat dinormalisasi menjadi bentuk yang lebih konsisten.

## 18. Mengapa Tidak Boleh Sembarangan Menghapus Repeated Characters?

Karena:

```text
"aaaa"
```

belum tentu noise.

Dan pada beberapa kata atau ekspresi, karakter berulang dapat menjadi bagian dari gaya bahasa pengguna.

Lebih penting lagi:

> preprocessing harus konsisten antara data training dan data inference.

Jika training menggunakan normalisasi tetapi API tidak menggunakannya, performa dapat berubah.

## 19. Tokenization

Tokenization adalah proses memecah teks menjadi unit yang disebut token.

Contoh:

```text
"Saya membutuhkan bantuan"
```

menjadi:

```text
["Saya", "membutuhkan", "bantuan"]
```

Token dapat berupa:

- kata,
- subword,
- karakter.

## 20. Word Tokenization

Contoh:

```python
text = "Saya membutuhkan bantuan"

tokens = text.split()

print(tokens)
```

Output:

```text
["Saya", "membutuhkan", "bantuan"]
```

Namun `.split()` terlalu sederhana untuk NLP modern.

## 21. Tokenisasi dengan NLTK

Contoh:

```python
from nltk.tokenize import word_tokenize

tokens = word_tokenize(text)
```

Namun untuk bahasa Indonesia, tokenizer harus dipilih dengan hati-hati.

## 22. Subword Tokenization

Ini sangat penting untuk memahami BERT.

Misalnya kata:

```text
kebanjiran
```

dapat dipecah menjadi subword tertentu oleh tokenizer model.

Konsepnya:

```text
Word
 ↓
Subword
 ↓
Token IDs
```

BERT tidak sekadar melakukan:

```text
1 kata = 1 token
```

Inilah salah satu alasan Anda perlu memahami tokenisasi sebelum masuk IndoBERT.

## 23. Stopword

Stopword adalah kata yang sering muncul tetapi biasanya memiliki informasi semantik yang relatif rendah untuk tugas tertentu.

Contoh:

```text
yang
dan
di
ke
dari
ini
itu
```

Contoh:

```text
"Saya membutuhkan bantuan dari pemerintah"
```

Setelah stopword removal:

```text
"membutuhkan bantuan pemerintah"
```

## 24. Apakah Stopword Removal Wajib?

**Tidak.**

Untuk:

```text
TF-IDF + SVM
```

stopword removal kadang bermanfaat.

Tetapi untuk:

```text
BERT / IndoBERT
```

Anda **tidak seharusnya otomatis melakukan stopword removal**.

Mengapa?

Karena BERT memanfaatkan konteks seluruh kalimat.

Contoh:

```text
"saya tidak membutuhkan bantuan"
```

Jika kata `tidak` dihapus:

```text
"saya membutuhkan bantuan"
```

Maknanya berubah secara drastis.

Untuk klasifikasi emergency, ini bisa menyebabkan kesalahan.

## 25. Negation Handling

Contoh:

```text
"Tidak membutuhkan bantuan"
```

berbeda dengan:

```text
"Membutuhkan bantuan"
```

Jangan melakukan preprocessing yang menghilangkan:

```text
tidak
bukan
belum
jangan
tanpa
```

tanpa analisis.

Untuk penelitian Anda, kata negasi sebaiknya dipertahankan.

## 26. Stemming

Stemming mengubah kata menjadi bentuk dasar atau stem.

Contoh:

```text
membantu
dibantu
membantukan
```

dapat diarahkan ke:

```text
bantu
```

Untuk bahasa Indonesia, salah satu library yang populer adalah:

```text
Sastrawi
```

## 27. Contoh Sastrawi

Install:

```bash
pip install Sastrawi
```

Kemudian:

```python
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

text = "Kami membutuhkan bantuan"

result = stemmer.stem(text)

print(result)
```

## 28. Kelebihan Stemming

Stemming dapat mengurangi variasi morfologi.

Misalnya:

```text
membantu
dibantu
membantukan
```

menjadi bentuk dasar yang lebih seragam.

Ini bisa membantu metode seperti:

```text
TF-IDF
Bag of Words
```

karena vocabulary menjadi lebih kecil.

## 29. Kekurangan Stemming

Stemming tidak selalu menghasilkan kata yang sempurna.

Selain itu, informasi morfologi dapat hilang.

Karena itu:

> stemming bukan sesuatu yang harus selalu dilakukan.

## 30. Stemming vs IndoBERT

Ini sangat penting untuk penelitian Anda.

Untuk:

```text
TF-IDF + SVM
```

Anda bisa melakukan:

```text
Cleaning
 ↓
Normalization
 ↓
Stemming
 ↓
TF-IDF
 ↓
SVM
```

Tetapi untuk:

```text
IndoBERT
```

jangan otomatis:

```text
Cleaning
 ↓
Stemming
 ↓
IndoBERT
```

Karena IndoBERT sudah menggunakan representasi kontekstual dan tokenizer subword.

## 31. Slang / Bahasa Tidak Formal

Media sosial penuh dengan bahasa informal.

Contoh:

```text
gak
ga
nggak
gk
tdk
```

Bisa dinormalisasi menjadi:

```text
tidak
```

Contoh:

```text
bgt → banget
udh → sudah
sdh → sudah
yg → yang
dr → dari
dgn → dengan
```

## 32. Slang Dictionary

Anda dapat membuat dictionary:

```python
slang_dict = {
    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "bgt": "banget",
    "udh": "sudah",
    "sdh": "sudah",
    "yg": "yang"
}
```

Kemudian:

```python
def normalize_slang(tokens):
    return [
        slang_dict.get(token, token)
        for token in tokens
    ]
```

## 33. Mengapa Normalisasi Slang Penting?

Dataset Anda kemungkinan akan memiliki:

```text
"Tolong bgt kami butuh bantuan"
```

dan:

```text
"Tolong banget kami membutuhkan bantuan"
```

Secara semantik:

```text
bgt ≈ banget
```

Normalisasi membantu mengurangi variasi.

## 34. Typo

Tweet dapat memiliki typo:

```text
bantuan
bntuan
bantuann
bantuaan
```

Masalahnya:

> Apakah typo harus diperbaiki?

Tidak selalu.

Automatic spelling correction dapat memperkenalkan kesalahan baru.

Untuk penelitian, Anda dapat membandingkan:

```text
Tanpa typo correction
```

vs.

```text
Dengan typo correction
```

Jika perbedaannya kecil, pipeline yang lebih sederhana biasanya lebih baik.

## 35. Named Entity

Dalam penelitian kebencanaan, entity sangat penting.

Contoh:

```text
"Rumah kami di Palu Barat terendam banjir."
```

Entity:

```text
Palu Barat → LOCATION
```

Contoh:

```text
"BPBD Palu"
```

```text
BPBD → ORGANIZATION
Palu → LOCATION
```

Jangan sembarangan menghapus entity.

Bahkan informasi lokasi dapat menjadi fitur penting dalam sistem monitoring bencana.

## 36. Location Information

Untuk penelitian Anda, lokasi dapat menjadi informasi yang sangat penting.

Contoh:

```text
"Tolong bantu kami di Desa Lolu"
```

Jika Anda menghapus:

```text
Desa Lolu
```

Anda kehilangan informasi penting.

Karena itu preprocessing harus mempertahankan entity yang relevan.

## 37. Contoh Preprocessing Social Media

Raw:

```text
"TOLONGGG 😭😭 @BPBD!!!
kami butuh bantuannnn di Desa Lolu,
air sdh setinggi dada!!! #BanjirPalu
https://example.com"
```

### Tahap 1 — URL

```text
"TOLONGGG 😭😭 @BPBD!!!
kami butuh bantuannnn di Desa Lolu,
air sdh setinggi dada!!! #BanjirPalu"
```

### Tahap 2 — Mention

Bisa:

```text
"TOLONGGG 😭😭
kami butuh bantuannnn di Desa Lolu,
air sdh setinggi dada!!! #BanjirPalu"
```

atau:

```text
"TOLONGGG 😭😭 [MENTION]
kami butuh bantuannnn di Desa Lolu,
air sdh setinggi dada!!! #BanjirPalu"
```

### Tahap 3 — Slang

```text
"tolonggg 😭😭
kami butuh bantuannnn di desa lolu,
air sudah setinggi dada!!! #banjirpalu"
```

### Tahap 4 — Repeated Characters

```text
"tolong 😭😭
kami butuh bantuan di desa lolu,
air sudah setinggi dada!!! #banjirpalu"
```

### Tahap 5 — Hashtag

Bisa menjadi:

```text
"tolong 😭😭 kami butuh bantuan di desa lolu,
air sudah setinggi dada!!! banjir palu"
```

Hasil akhir tetap harus mempertahankan informasi:

```text
tolong
bantuan
desa lolu
air
setinggi dada
banjir
palu
```

## 38. Pipeline untuk Penelitian Anda

Saya menyarankan Anda memiliki **dua pipeline preprocessing**.

### Pipeline A — Baseline TF-IDF

```text
Raw Tweet
   ↓
Remove URL
   ↓
Handle Mention
   ↓
Handle Hashtag
   ↓
Unicode Normalization
   ↓
Lowercase
   ↓
Normalize Slang
   ↓
Normalize Repeated Characters
   ↓
Tokenization
   ↓
Optional Stopword Removal
   ↓
Optional Stemming
   ↓
TF-IDF
   ↓
SVM / Logistic Regression
```

## 39. Pipeline B — IndoBERT

Untuk IndoBERT, pipeline harus lebih minimal.

```text
Raw Tweet
   ↓
Basic Cleaning
   ↓
Handle URL
   ↓
Handle Noise yang jelas
   ↓
Preserve Semantic Information
   ↓
IndoBERT Tokenizer
   ↓
Input IDs
   ↓
Attention Mask
   ↓
IndoBERT
```

Jangan melakukan secara otomatis:

```text
❌ Stopword removal
❌ Stemming
❌ Menghapus semua punctuation
❌ Menghapus semua angka
❌ Menghapus semua emoji
❌ Mengubah semua teks secara agresif
```

## 40. Mengapa Pipeline IndoBERT Berbeda?

Ada perbedaan fundamental.

### TF-IDF

Model melihat:

```text
kata → frequency
```

Sehingga normalisasi vocabulary sangat penting.

### IndoBERT

Model melihat:

```text
token
+
context
+
position
+
attention
```

Jadi terlalu banyak preprocessing dapat menghilangkan informasi yang dibutuhkan model.

## 41. Contoh Kesalahan Preprocessing

Original:

```text
"Saya TIDAK butuh bantuan"
```

Jika stopword removal dilakukan secara agresif:

```text
"butuh bantuan"
```

Model dapat menyimpulkan:

```text
Emergency
```

padahal:

```text
Non-Emergency
```

Ini contoh bahwa preprocessing dapat menyebabkan **semantic distortion**.

## 42. Data Leakage dalam Preprocessing

Ini penting untuk penelitian.

Misalnya Anda melakukan:

```text
Dataset
 ↓
TF-IDF
 ↓
Train/Test Split
```

Ini dapat menyebabkan vocabulary/statistik TF-IDF dipelajari dari data test.

Yang benar:

```text
Dataset
 ↓
Train/Test Split
 ↓
TF-IDF.fit(train)
 ↓
TF-IDF.transform(train)
TF-IDF.transform(test)
```

Dengan Scikit-learn, gunakan `Pipeline`.

Contoh:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", LinearSVC())
])

model.fit(X_train, y_train)
```

Dengan demikian proses transformasi lebih aman dari leakage.

## 43. Preprocessing dan Train/Test Consistency

Hal yang sangat penting:

Training:

```text
Raw Tweet
 ↓
Preprocessing
 ↓
Model
```

Inference:

```text
Tweet baru
 ↓
Preprocessing yang SAMA
 ↓
Model
```

Jangan sampai:

```text
Training:
clean → normalize → tokenize
```

sedangkan API:

```text
raw → model
```

Pipeline harus konsisten.

## 44. Membuat Fungsi Preprocessing

Contoh sederhana:

```python
import re
import unicodedata

def clean_text(text):
    text = unicodedata.normalize("NFKC", text)

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text
```

Kemudian:

```python
text = """
Tolong @BPBD bantu kami!!!
https://example.com
"""

print(clean_text(text))
```

Output:

```text
Tolong bantu kami!!!
```

## 45. Membuat Pipeline yang Lebih Terstruktur

Untuk penelitian, lebih baik fungsi dipisahkan:

```python
def normalize_unicode(text):
    pass

def remove_url(text):
    pass

def handle_mention(text):
    pass

def normalize_slang(text):
    pass

def normalize_repeated_chars(text):
    pass

def clean_whitespace(text):
    pass

def preprocess_text(text):
    text = normalize_unicode(text)
    text = remove_url(text)
    text = handle_mention(text)
    text = normalize_slang(text)
    text = normalize_repeated_chars(text)
    text = clean_whitespace(text)

    return text
```

Keuntungannya:

- mudah diuji,
- mudah diperbaiki,
- mudah dibandingkan,
- mudah digunakan kembali pada API.

## 46. Eksperimen Preprocessing

Untuk penelitian, jangan hanya membuat satu preprocessing.

Buat eksperimen.

### Experiment A — Minimal

```text
URL removal
Mention handling
Whitespace
```

### Experiment B — Normalization

```text
A
+
Lowercase
+
Slang normalization
+
Repeated character normalization
```

### Experiment C — Aggressive

```text
B
+
Stopword removal
+
Stemming
```

Kemudian:

```text
 Experiment
     ↓
   Train
     ↓
 Evaluate
     ↓
 Compare F1
```

Misalnya:

| Preprocessing | Accuracy | Recall | F1 |
|---|---:|---:|---:|
| Minimal | 0.89 | 0.87 | 0.88 |
| Normalization | 0.91 | 0.90 | 0.90 |
| Aggressive | 0.88 | 0.84 | 0.85 |

Jika hasilnya seperti ini, Anda memiliki bukti empiris bahwa preprocessing tertentu lebih baik.

## 47. Preprocessing sebagai Bagian Metodologi Penelitian

Dalam paper atau tesis Anda nantinya, bagian preprocessing dapat ditulis sebagai:

```text
Raw Social Media Data
        ↓
Data Cleaning
        ↓
URL Handling
        ↓
Mention Handling
        ↓
Text Normalization
        ↓
Tokenization
        ↓
Model-Specific Processing
        ↓
Classification
```

Dan Anda dapat menjelaskan alasan setiap tahap.

## 48. Checklist Preprocessing untuk Dataset Anda

Sebelum masuk training, periksa:

```text
□ Apakah ada duplicate tweet?
□ Apakah ada URL?
□ Apakah ada mention?
□ Apakah ada hashtag?
□ Apakah ada emoji?
□ Apakah ada slang?
□ Apakah ada typo?
□ Apakah ada repeated characters?
□ Apakah ada HTML?
□ Apakah ada bahasa campuran?
□ Apakah ada angka?
□ Apakah ada lokasi?
□ Apakah ada negation?
□ Apakah ada duplicate data?
□ Apakah ada missing text?
□ Apakah class imbalance?
```

## 49. Hal yang Harus Dikuasai Sebelum Lanjut

### Konsep

```text
✓ Raw text
✓ Cleaning
✓ Case folding
✓ Normalization
✓ Tokenization
✓ Stopword
✓ Stemming
✓ Slang normalization
✓ Emoji
✓ URL
✓ Mention
✓ Hashtag
✓ Negation
✓ Social media noise
```

### Python

Anda harus bisa membuat:

```python
clean_text()
normalize_text()
tokenize_text()
```

dan menggabungkannya:

```python
preprocess_text()
```

### Penelitian

Anda harus bisa menjawab:

> **Mengapa tahap preprocessing ini dilakukan?**

Dan yang lebih penting:

> **Mengapa tahap preprocessing tertentu tidak dilakukan pada IndoBERT?**

Jika Anda sudah bisa menjawab dua pertanyaan tersebut, Anda sudah memiliki fondasi preprocessing yang cukup kuat untuk masuk ke:

```text
Text Representation
        ↓
      TF-IDF
        ↓
NLP Classification
        ↓
  Word Embedding
        ↓
   Transformer
        ↓
     IndoBERT
```

## 50. Latihan Utama

Buat satu notebook atau folder pembelajaran:

```text
02-text-preprocessing/
│
├── 01_case_folding.ipynb
├── 02_cleaning.ipynb
├── 03_tokenization.ipynb
├── 04_stopword.ipynb
├── 05_stemming.ipynb
├── 06_slang_normalization.ipynb
├── 07_social_media_preprocessing.ipynb
└── 08_complete_pipeline.ipynb
```

Gunakan contoh tweet bencana:

```text
"TOLONGGG 😭😭 @BPBD bantu kami!!!
rumah kami kebanjirannnn 😭
air sdh setinggi dada #BanjirPalu
```

Target akhirnya:

```text
Raw Tweet
      ↓
┌──────────────────────┐
│ Preprocessing        │
│                      │
│ URL                  │
│ Mention              │
│ Hashtag              │
│ Slang                │
│ Repeated Characters  │
│ Unicode              │
│ Whitespace            │
└──────────┬───────────┘
           ↓
   Normalized Tweet
           ↓
    ┌──────┴──────┐
    ↓             ↓
 TF-IDF        IndoBERT
    ↓             ↓
 SVM          Tokenizer
    ↓             ↓
Baseline      Fine-tuning
```

## 51. Kesimpulan

Text preprocessing bukan sekadar proses "membersihkan teks". Dalam penelitian NLP, preprocessing merupakan bagian dari **experimental design** karena setiap transformasi dapat memengaruhi informasi yang diterima model.

Untuk penelitian deteksi tweet permintaan bantuan darurat, prinsip utama yang perlu digunakan adalah:

1. Kurangi noise.
2. Pertahankan informasi semantik.
3. Pertahankan informasi lokasi yang relevan.
4. Jangan menghapus negasi secara sembarangan.
5. Jangan menghapus angka secara otomatis.
6. Jangan melakukan stemming dan stopword removal secara otomatis pada IndoBERT.
7. Pisahkan pipeline baseline TF-IDF dan pipeline IndoBERT.
8. Hindari data leakage.
9. Gunakan preprocessing yang konsisten saat training dan inference.
10. Validasi keputusan preprocessing melalui eksperimen, bukan asumsi.

Dengan prinsip tersebut, preprocessing Anda dapat menjadi bagian metodologi penelitian yang dapat dipertanggungjawabkan, bukan sekadar tahap cleaning sebelum training model.
