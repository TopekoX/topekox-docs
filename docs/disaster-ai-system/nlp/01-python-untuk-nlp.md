# Python untuk NLP

## 🎯 Tujuan Pembelajaran

Setelah menyelesaikan Level ini, Anda diharapkan mampu:

-   Memahami bagaimana Python merepresentasikan teks.
-   Memanipulasi `string`.
-   Menggunakan `list`, `tuple`, `dictionary`, dan `set` untuk data
    teks.
-   Membuat function untuk preprocessing teks.
-   Membaca dataset teks.
-   Memahami JSON.
-   Menggunakan Regular Expression (`re`).
-   Membuat preprocessing sederhana untuk teks Indonesia.
-   Menyiapkan fondasi untuk **Text Preprocessing** pada Level 2.

------------------------------------------------------------------------

## 1 Apa Hubungan Python dengan NLP?

**NLP (Natural Language Processing)** bekerja dengan data berupa bahasa
manusia.

Contoh:

``` text
"Tolong bantu kami, rumah kami terendam banjir"
```

Masalahnya, komputer tidak memahami kalimat tersebut seperti manusia.

Kita perlu mengubah dan memproses teks:

``` text
Teks
 ↓
Python String
 ↓
Cleaning
 ↓
Tokenization
 ↓
Numerical Representation
 ↓
Machine Learning
```

Python berfungsi sebagai alat untuk melakukan seluruh proses tersebut.

Contoh sederhana:

``` python
text = "Tolong bantu kami"

print(text)
```

Output:

``` text
Tolong bantu kami
```

------------------------------------------------------------------------

## 2 Python String

Dalam NLP, `string` adalah salah satu objek yang paling sering
digunakan.

``` python
text = "Tolong bantu kami"
```

Kita dapat memeriksa tipe datanya:

``` python
print(type(text))
```

Output:

``` text
<class 'str'>
```

### Menghitung panjang teks

``` python
text = "Tolong bantu kami"

print(len(text))
```

`len()` menghitung jumlah karakter.

### Mengakses karakter

``` python
text = "Python"

print(text[0])
print(text[1])
print(text[2])
```

Output:

``` text
P
y
t
```

Ingat bahwa index Python dimulai dari:

``` text
P  y  t  h  o  n
0  1  2  3  4  5
```

------------------------------------------------------------------------

## 3 String Slicing

Kita bisa mengambil sebagian teks.

``` python
text = "Natural Language Processing"

print(text[0:7])
```

Output:

``` text
Natural
```

Contoh lain:

``` python
text = "Natural Language Processing"

print(text[8:16])
```

Output:

``` text
Language
```

Slicing akan sangat berguna ketika melakukan manipulasi teks.

------------------------------------------------------------------------

## 4 Mengubah Huruf

NLP sering membutuhkan **case normalization**.

Contoh:

``` python
text = "Tolong Bantu Kami"
```

### Lowercase

``` python
print(text.lower())
```

Output:

``` text
tolong bantu kami
```

### Uppercase

``` python
print(text.upper())
```

Output:

``` text
TOLONG BANTU KAMI
```

### Capitalize

``` python
print(text.capitalize())
```

Output:

``` text
Tolong bantu kami
```

------------------------------------------------------------------------

## 5 Menghapus Spasi

Menghapus spasi di awal dan akhir string.

Gunakan:

``` python
text = "   Tolong bantu kami   "

print(text.strip())
```

Output:

``` text
Tolong bantu kami
```

`strip()` berguna ketika dataset memiliki spasi yang tidak diperlukan.

Contoh:

``` text
"   banjir di Palu   "
```

menjadi:

``` text
"banjir di Palu"
```

------------------------------------------------------------------------

## 6 Replace

Kita dapat mengganti bagian teks.

``` python
text = "Saya tinggal di Jakarta"

text = text.replace("Jakarta", "Palu")

print(text)
```

Output:

``` text
Saya tinggal di Palu
```

Dalam NLP, `replace()` sering digunakan untuk normalisasi sederhana.

Contoh:

``` python
text = "Saya tidak bisa pergi"

text = text.replace("tidak", "gak")

print(text)
```

------------------------------------------------------------------------

## 7 Split

Ini salah satu operasi **sangat penting dalam NLP**.

``` python
text = "Saya membutuhkan bantuan"

words = text.split()

print(words)
```

Output:

``` python
['Saya', 'membutuhkan', 'bantuan']
```

Perhatikan bahwa string berubah menjadi `list`.

``` python
print(type(words))
```

Output:

``` text
<class 'list'>
```

Secara sederhana:

``` text
"Saya membutuhkan bantuan"
              ↓
          split()
              ↓
["Saya", "membutuhkan", "bantuan"]
```

Ini adalah bentuk sederhana dari **tokenization**.

> Nanti kita akan belajar tokenization yang sebenarnya pada Materi yang akan datang.

------------------------------------------------------------------------

## 8 Split dengan Delimiter

Misalnya:

``` python
text = "banjir,palu,sulawesi"
```

Kita bisa menggunakan:

``` python
data = text.split(",")

print(data)
```

Output:

``` python
['banjir', 'palu', 'sulawesi']
```

Contoh lain:

``` python
text = "banjir|palu|sulawesi"

data = text.split("|")

print(data)
```

------------------------------------------------------------------------

## 9 Join

Kebalikan dari `split()` adalah `join()`.

``` python
words = ["Saya", "membutuhkan", "bantuan"]

text = " ".join(words)

print(text)
```

Output:

``` text
Saya membutuhkan bantuan
```

Konsep:

``` text
List
 ↓
join()
 ↓
String
```

------------------------------------------------------------------------

## 10 List untuk Data Teks

Dalam NLP kita sering memiliki banyak dokumen.

Misalnya:

``` python
texts = [
    "Saya membutuhkan bantuan",
    "Terjadi banjir di desa kami",
    "Cuaca hari ini sangat cerah"
]
```

Kita bisa mengambil teks pertama:

``` python
print(texts[0])
```

Output:

``` text
Saya membutuhkan bantuan
```

Mengambil teks kedua:

``` python
print(texts[1])
```

### Loop pada teks

``` python
for text in texts:
    print(text)
```

Output:

``` text
Saya membutuhkan bantuan
Terjadi banjir di desa kami
Cuaca hari ini sangat cerah
```

Ini akan sangat sering digunakan ketika melakukan preprocessing dataset.

------------------------------------------------------------------------

## 11 Dictionary untuk NLP

Dictionary digunakan untuk menyimpan pasangan:

``` text
key → value
```

Contoh:

``` python
tweet = {
    "text": "Tolong bantu kami",
    "label": "emergency"
}
```

Ambil teks:

``` python
print(tweet["text"])
```

Output:

``` text
Tolong bantu kami
```

Ambil label:

``` python
print(tweet["label"])
```

Output:

``` text
emergency
```

### Dataset sederhana

Kita bisa memiliki:

``` python
dataset = [
    {
        "text": "Tolong bantu kami",
        "label": "emergency"
    },
    {
        "text": "Cuaca hari ini cerah",
        "label": "non-emergency"
    }
]
```

Kemudian:

``` python
for data in dataset:
    print(data["text"])
```

Output:

``` text
Tolong bantu kami
Cuaca hari ini cerah
```

Ini merupakan gambaran sederhana struktur dataset NLP.

------------------------------------------------------------------------

## 12 Set

`set` berguna ketika kita ingin mendapatkan nilai unik.

Contoh:

``` python
words = [
    "banjir",
    "bantuan",
    "banjir",
    "rumah",
    "bantuan"
]

unique_words = set(words)

print(unique_words)
```

Hasilnya hanya menyimpan kata unik.

Konsep ini penting ketika nanti memahami:

> **Vocabulary**

Misalnya:

``` text
Saya butuh bantuan
Saya butuh makanan
```

Vocabulary-nya:

``` text
Saya
butuh
bantuan
makanan
```

------------------------------------------------------------------------

## 13 Function untuk NLP

Daripada menulis preprocessing berkali-kali:

``` python
text.lower()
text.strip()
text.replace(...)
```

lebih baik membuat function.

``` python
def clean_text(text):
    text = text.lower()
    text = text.strip()

    return text
```

Gunakan:

``` python
text = "   TOLONG BANTU KAMI   "

result = clean_text(text)

print(result)
```

Output:

``` text
tolong bantu kami
```

------------------------------------------------------------------------

## 14 Function dengan Beberapa Proses

Kita bisa membuat preprocessing sederhana:

``` python
def clean_text(text):
    text = text.lower()
    text = text.strip()
    text = text.replace("!", "")
    text = text.replace("?", "")

    return text
```

Kemudian:

``` python
text = "TOLONG BANTU KAMI!!!"

print(clean_text(text))
```

Output:

``` text
tolong bantu kami
```

Inilah awal dari konsep:

``` text
Preprocessing Pipeline
```

------------------------------------------------------------------------

## 15 Conditional

Kita juga harus memahami `if`.

Contoh:

``` python
text = "Tolong bantu kami"

if "bantuan" in text.lower():
    print("Mengandung kata bantuan")
else:
    print("Tidak ditemukan")
```

Output:

``` text
Tidak ditemukan
```

Contoh:

``` python
text = "Kami membutuhkan bantuan"

if "bantuan" in text.lower():
    print("Kemungkinan emergency")
```

Output:

``` text
Kemungkinan emergency
```

Namun hati-hati:

> Ini **belum Machine Learning**.

Kita hanya menggunakan rule sederhana.

------------------------------------------------------------------------

## 16 Regular Expression

Sekarang bagian yang sangat penting untuk NLP.

Python menyediakan library:

``` python
import re
```

Regular Expression digunakan untuk mencari pola dalam teks.

Misalnya:

``` text
https://example.com
```

Kita ingin menghapus URL.

### Contoh sederhana

``` python
import re

text = "Kunjungi https://example.com untuk informasi"

text = re.sub(r"https?://\S+", "", text)

print(text)
```

Output:

``` text
Kunjungi  untuk informasi
```

------------------------------------------------------------------------

## 17 Menghapus Mention

Data Twitter/X sering memiliki:

``` text
@BPBD_Sulteng
```

Kita dapat menggunakan:

``` python
text = "Tolong @BPBD_Sulteng bantu kami"

text = re.sub(r"@\w+", "", text)

print(text)
```

Output:

``` text
Tolong  bantu kami
```

------------------------------------------------------------------------

## 18 Menghapus Hashtag

Misalnya:

``` text
#BanjirPalu
```

Gunakan:

``` python
text = "Terjadi banjir #BanjirPalu"

text = re.sub(r"#\w+", "", text)

print(text)
```

Output:

``` text
Terjadi banjir
```

Tetapi dalam penelitian NLP, **jangan selalu menghapus hashtag**.

Misalnya:

``` text
#PrayForPalu
```

bisa memiliki informasi penting.

Jadi preprocessing harus disesuaikan dengan tujuan penelitian.

------------------------------------------------------------------------

## 19 Menghapus Punctuation

Gunakan:

``` python
import string

text = "Tolong bantu kami!!!"

text = text.translate(
    str.maketrans("", "", string.punctuation)
)

print(text)
```

Output:

``` text
Tolong bantu kami
```

------------------------------------------------------------------------

## 20 JSON

Data NLP sering disimpan dalam JSON.

Contoh:

``` json
{
    "text": "Tolong bantu kami",
    "label": "emergency"
}
```

Python memiliki library:

``` python
import json
```

Contoh:

``` python
data = {
    "text": "Tolong bantu kami",
    "label": "emergency"
}

json_data = json.dumps(data)

print(json_data)
```

------------------------------------------------------------------------

## 21 Membaca File JSON

Misalnya kita memiliki:

``` text
dataset.json
```

Kita bisa membaca:

``` python
import json

with open("dataset.json", "r") as file:
    data = json.load(file)

print(data)
```

Ini akan berguna ketika nanti mengambil dataset dari API atau menyimpan
dataset hasil scraping.

------------------------------------------------------------------------

## 22 Membaca CSV

Dataset NLP sering berbentuk:

``` text
dataset.csv
```

Misalnya:

``` csv
text,label
"Tolong bantu kami",emergency
"Cuaca hari ini cerah",non-emergency
```

Dengan Pandas:

``` python
import pandas as pd

df = pd.read_csv("dataset.csv")

print(df.head())
```

Output kira-kira:

``` text
                              text           label
0              Tolong bantu kami      emergency
1          Cuaca hari ini cerah      non-emergency
```

------------------------------------------------------------------------

## 23 Mengambil Kolom Teks

``` python
texts = df["text"]
```

Label:

``` python
labels = df["label"]
```

Kemudian:

``` python
print(texts.head())
```

------------------------------------------------------------------------

## 24 Mini NLP Pipeline

Sekarang gabungkan beberapa materi yang sudah dipelajari.

``` python
import re

def clean_text(text):

    # lowercase
    text = text.lower()

    # hapus URL
    text = re.sub(r"https?://\S+", "", text)

    # hapus mention
    text = re.sub(r"@\w+", "", text)

    # hapus hashtag
    text = re.sub(r"#\w+", "", text)

    # hapus karakter selain huruf
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # hapus spasi berlebih
    text = re.sub(r"\s+", " ", text)

    return text.strip()
```

Gunakan:

``` python
text = """
TOLONG @BPBD!!! Bantu kami 😭
Rumah terendam banjir!!!
https://example.com
"""

result = clean_text(text)

print(result)
```

Hasil:

``` text
tolong bantu kami rumah terendam banjir
```

Ini sudah menjadi **mini text preprocessing pipeline**.

------------------------------------------------------------------------

## 25 Pipeline Python → NLP

Anda perlu memahami alur berikut:

``` text
Raw Text
   │
   ▼
Python String
   │
   ├── lower()
   ├── strip()
   ├── replace()
   ├── split()
   └── regex
   │
   ▼
Clean Text
   │
   ▼
Token
   │
   ▼
Numerical Representation
   │
   ▼
Machine Learning
```

------------------------------------------------------------------------

## 🧪 PRAKTIK LEVEL 0

### Project: Simple Tweet Cleaner

Input:

``` python
tweet = """
TOLONG @BPBD_SULTENG bantu kami!!!
Banjir sudah masuk ke rumah 😭😭

Info: https://contoh.com

#BanjirPalu #Darurat
"""
```

Target output:

``` text
tolong bantu kami banjir sudah masuk ke rumah
```

Buat function:

``` python
def clean_tweet(text):
    ...
```

Function tersebut minimal melakukan:

``` text
1. lowercase
2. remove URL
3. remove mention
4. remove hashtag
5. remove punctuation
6. remove emoji
7. remove extra whitespace
```

------------------------------------------------------------------------

## 🧪 PRAKTIK 2 --- Dataset

Buat dataset sederhana:

``` python
dataset = [
    "Tolong bantu kami, rumah terendam banjir",
    "Cuaca hari ini sangat cerah",
    "Mohon bantuan, kami terjebak banjir",
    "Saya sedang makan di rumah",
    "Jalan desa kami terputus akibat longsor"
]
```

Kemudian gunakan:

``` python
for text in dataset:
    print(clean_tweet(text))
```

Target:

``` text
tolong bantu kami rumah terendam banjir
cuaca hari ini sangat cerah
mohon bantuan kami terjebak banjir
saya sedang makan di rumah
jalan desa kami terputus akibat longsor
```

------------------------------------------------------------------------

## 🧠 Yang Harus Anda Kuasai Sebelum LEVEL Selanjutnya

Jangan lanjut sebelum Anda cukup nyaman dengan:

### Python

-   [ ] String
-   [ ] Indexing
-   [ ] Slicing
-   [ ] `lower()`
-   [ ] `upper()`
-   [ ] `strip()`
-   [ ] `replace()`
-   [ ] `split()`
-   [ ] `join()`
-   [ ] List
-   [ ] Dictionary
-   [ ] Set
-   [ ] Loop
-   [ ] Function
-   [ ] `if`
-   [ ] File handling
-   [ ] JSON
-   [ ] Pandas CSV

### NLP-oriented Python

-   [ ] `re`
-   [ ] Regex pattern
-   [ ] Cleaning URL
-   [ ] Cleaning mention
-   [ ] Cleaning hashtag
-   [ ] Cleaning punctuation
-   [ ] Normalisasi whitespace
-   [ ] Membuat preprocessing function

------------------------------------------------------------------------

## 🎯 Mini Project Akhir

Saya sarankan Anda membuat program:

``` text
              DATASET TWEET
                    │
                    ▼
             clean_tweet()
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
     Lowercase     URL       Mention
        │           │           │
        └───────────┼───────────┘
                    ▼
                Hashtag
                    │
                    ▼
               Punctuation
                    │
                    ▼
              Clean Dataset
                    │
                    ▼
               dataset.csv
```

Output akhirnya berupa:

``` text
raw_text,clean_text
"TOLONG @BPBD!!!","tolong"
"Rumah kami terendam banjir","rumah kami terendam banjir"
```

------------------------------------------------------------------------

## 🚀 Setelah ini

Setelah mini-project ini selesai, Anda siap masuk:

**Fundamental NLP**

Materi berikutnya:

-   Apa itu NLP?
-   Corpus
-   Document
-   Sentence
-   Token
-   Vocabulary
-   Lexical analysis
-   NLP pipeline
-   Text classification
-   Sentiment analysis
-   Spam detection
-   Named Entity Recognition
-   Text similarity
-   Text clustering
-   Machine Translation
-   Question Answering
-   Text Generation

Urutan ini akan menjadi fondasi sebelum masuk ke **Text Preprocessing →
TF-IDF → Machine Learning → Transformer → BERT → IndoBERT**.
