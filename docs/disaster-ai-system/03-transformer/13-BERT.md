# BERT (Bidirectional Encoder Representations from Transformers)

> **Level:** Menengah  
> **Prasyarat:** Fundamental NLP, tokenization, embeddings, attention mechanism, dan Transformer  
> **Tujuan:** Memahami konsep BERT, cara kerja tokenizer, representasi input, pre-training, serta penggunaan BERT untuk klasifikasi teks. Materi ini menjadi jembatan sebelum mempelajari IndoBERT dan fine-tuning untuk penelitian.

---

## Daftar Isi

1. [Tujuan Pembelajaran](#1-tujuan-pembelajaran)
2. [Apa Itu BERT?](#2-apa-itu-bert)
3. [Mengapa BERT Dibutuhkan?](#3-mengapa-bert-dibutuhkan)
4. [BERT dan Transformer Encoder](#4-bert-dan-transformer-encoder)
5. [Bidirectional Context](#5-bidirectional-context)
6. [Token Khusus BERT](#6-token-khusus-bert)
7. [Tokenization dan Input BERT](#7-tokenization-dan-input-bert)
8. [Embedding dalam BERT](#8-embedding-dalam-bert)
9. [Pre-training BERT](#9-pre-training-bert)
10. [Fine-tuning BERT](#10-fine-tuning-bert)
11. [Praktik 1: Tokenizer dengan Hugging Face](#11-praktik-1-tokenizer-dengan-hugging-face)
12. [Praktik 2: Mengambil Representasi BERT](#12-praktik-2-mengambil-representasi-bert)
13. [Praktik 3: Klasifikasi Teks dengan Pipeline](#13-praktik-3-klasifikasi-teks-dengan-pipeline)
14. [BERT untuk Klasifikasi Permintaan Bantuan](#14-bert-untuk-klasifikasi-permintaan-bantuan)
15. [BERT vs TF-IDF dan Word2Vec](#15-bert-vs-tf-idf-dan-word2vec)
16. [Kelebihan dan Keterbatasan BERT](#16-kelebihan-dan-keterbatasan-bert)
17. [Kesalahan Umum](#17-kesalahan-umum)
18. [Latihan](#18-latihan)
19. [Mini Project](#19-mini-project)
20. [Checklist Kompetensi](#20-checklist-kompetensi)

---

## 1. Tujuan Pembelajaran

Setelah mempelajari materi ini, Anda diharapkan mampu:

- Menjelaskan BERT dan hubungannya dengan Transformer.
- Menjelaskan arti representasi bidirectional.
- Mengenali token khusus seperti `[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`, dan `[UNK]`.
- Memahami `input_ids`, `attention_mask`, dan `token_type_ids`.
- Menjelaskan perbedaan pre-training dan fine-tuning.
- Menggunakan tokenizer dan model BERT melalui Hugging Face Transformers.
- Memahami bagaimana BERT digunakan untuk klasifikasi teks.
- Menjelaskan hubungan BERT dengan IndoBERT untuk teks bahasa Indonesia.

---

## 2. Apa Itu BERT?

**BERT** adalah singkatan dari **Bidirectional Encoder Representations from Transformers**. BERT merupakan model bahasa yang menggunakan tumpukan **Transformer Encoder** untuk menghasilkan representasi teks yang mempertimbangkan konteks.

Misalnya kata *bisa* pada dua kalimat berikut:

- `Ular itu memiliki bisa.`
- `Saya bisa mengerjakan tugas.`

Kata *bisa* memiliki makna berbeda. Model berbasis konteks dapat mempertimbangkan kata-kata di sekitarnya untuk membentuk representasi yang berbeda.

Secara umum, alur BERT adalah:

```text
Teks
  ↓
Tokenizer
  ↓
Token IDs + Attention Mask
  ↓
Embedding
  ↓
Transformer Encoder
  ↓
Contextual Representations
  ↓
Task-specific Head (misalnya klasifikasi)
  ↓
Prediksi
```

BERT bukan aplikasi chatbot dan bukan model generatif autoregresif seperti GPT. BERT terutama dirancang sebagai encoder untuk memahami dan merepresentasikan teks, meskipun representasinya dapat digunakan dalam berbagai tugas NLP.

### Contoh tugas yang dapat menggunakan BERT

- Klasifikasi teks
- Analisis sentimen
- Pengenalan entitas bernama (NER)
- Klasifikasi pasangan kalimat
- Ekstraksi jawaban dari teks
- Klasifikasi topik

---

## 3. Mengapa BERT Dibutuhkan?

### 3.1 Keterbatasan representasi sederhana

Pada TF-IDF, setiap dokumen direpresentasikan berdasarkan bobot kata. TF-IDF sangat berguna sebagai baseline, tetapi tidak secara langsung memahami makna kontekstual sebuah kata.

Contoh:

```text
Kalimat A: Saya melihat bank di tepi sungai.
Kalimat B: Saya pergi ke bank untuk menabung.
```

Kata `bank` muncul di dua kalimat, tetapi konteksnya berbeda. TF-IDF terutama merepresentasikan frekuensi dan bobot kata, sedangkan BERT membentuk representasi berdasarkan konteks token di dalam kalimat.

### 3.2 Dari fitur buatan ke representasi kontekstual

```text
TF-IDF:
Teks → bobot token → classifier

BERT:
Teks → subword tokenization → Transformer Encoder
     → representasi kontekstual → classifier
```

BERT tidak otomatis selalu lebih baik. Kinerjanya harus diuji pada dataset dan dibandingkan dengan baseline yang masuk akal.

---

## 4. BERT dan Transformer Encoder

BERT menggunakan bagian **encoder** dari arsitektur Transformer. BERT asli tidak menggunakan Transformer Decoder untuk menghasilkan teks secara autoregresif.

Satu blok Transformer Encoder pada umumnya mencakup:

1. Multi-Head Self-Attention
2. Residual connection dan Layer Normalization
3. Feed-Forward Network
4. Residual connection dan Layer Normalization

Secara konseptual:

```text
Representasi input
       ↓
Multi-Head Self-Attention
       ↓
Residual + Layer Normalization
       ↓
Feed-Forward Network
       ↓
Residual + Layer Normalization
       ↓
Representasi yang diperbarui
```

BERT menumpuk beberapa blok encoder. Setiap lapisan memperbarui representasi token dengan informasi kontekstual dari token lain yang dapat diperhatikannya.

### Self-attention secara ringkas

Self-attention membentuk tiga proyeksi dari representasi input:

- **Query (Q):** informasi yang digunakan untuk mencari konteks relevan.
- **Key (K):** informasi untuk menghitung kecocokan dengan query.
- **Value (V):** informasi yang digabungkan berdasarkan bobot perhatian.

Formula scaled dot-product attention:

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
\]

BERT menggunakan multi-head attention sehingga model dapat mempelajari beberapa pola hubungan token secara paralel.

---

## 5. Bidirectional Context

Istilah **bidirectional** berarti representasi token dibentuk dengan mempertimbangkan konteks dari sisi kiri dan kanan pada urutan input, bukan hanya token sebelumnya.

Contoh:

```text
"Tolong bantu warga yang terdampak banjir"
```

Untuk membentuk representasi `terdampak`, model dapat mempertimbangkan kata-kata di sebelah kiri (`warga`, `yang`) dan kanan (`banjir`).

Penting: bidirectional bukan berarti model menjalankan dua model terpisah, satu dari kiri dan satu dari kanan. BERT memakai self-attention pada input sehingga token dapat mengakses konteks dari kedua arah.

---

## 6. Token Khusus BERT

Tokenizer BERT menggunakan token khusus untuk menandai struktur input.

| Token | Fungsi umum |
|---|---|
| `[CLS]` | Token awal; representasinya sering digunakan sebagai ringkasan input untuk klasifikasi. |
| `[SEP]` | Pemisah atau penanda akhir segmen teks. |
| `[MASK]` | Token yang digunakan dalam objective Masked Language Modeling. |
| `[PAD]` | Mengisi posisi kosong agar panjang input dalam satu batch seragam. |
| `[UNK]` | Menandai token yang tidak dikenal oleh tokenizer tertentu. |

Contoh konseptual satu kalimat:

```text
Teks:
Tolong bantu warga

Input token:
[CLS] Tolong bantu warga [SEP]
```

Untuk dua kalimat:

```text
[CLS] Kalimat pertama [SEP] Kalimat kedua [SEP]
```

Detail token dan perilaku tokenizer dapat berbeda antar-model. Periksa tokenizer yang sesuai dengan checkpoint yang digunakan.

---

## 7. Tokenization dan Input BERT

### 7.1 Word tokenization vs subword tokenization

Tokenizer BERT umumnya menggunakan pendekatan subword. Sebuah kata dapat dipertahankan sebagai satu token atau dipecah menjadi beberapa subword sesuai kosakata tokenizer.

Contoh ilustratif:

```text
Kata:
membutuhkan

Kemungkinan hasil subword:
["membutuhkan"]
```

Kata lain mungkin terpecah menjadi beberapa bagian. **Jangan menganggap hasil ilustratif ini sebagai output pasti**; hasil sebenarnya bergantung pada tokenizer dan checkpoint.

Subword membantu model menangani kata yang tidak seluruhnya terdapat dalam kosakata.

### 7.2 Tiga komponen input yang umum

- `input_ids`: ID numerik untuk setiap token.
- `attention_mask`: biasanya `1` untuk token nyata dan `0` untuk padding.
- `token_type_ids`: menandai segmen kalimat pada model yang menggunakannya; tidak selalu diperlukan atau dikembalikan oleh semua tokenizer.

Contoh bentuk data:

```python
{
    "input_ids": [101, 1234, 5678, 102],
    "attention_mask": [1, 1, 1, 1]
}
```

**Catatan:** angka di atas hanya ilustrasi, bukan ID token yang dijamin benar untuk suatu kalimat atau model.

### 7.3 Padding dan truncation

- **Padding:** menambahkan token `[PAD]` agar urutan dalam batch memiliki panjang yang sama.
- **Truncation:** memotong urutan yang melebihi batas panjang yang ditentukan.

Keduanya penting saat memproses banyak teks sekaligus.

---

## 8. Embedding dalam BERT

Pada model BERT klasik, representasi input dibentuk dengan menggabungkan tiga jenis embedding:

1. **Token embedding:** identitas token.
2. **Position embedding:** posisi token dalam urutan.
3. **Segment/token-type embedding:** segmen teks, terutama untuk input pasangan kalimat.

Secara konseptual:

\[
E_{\text{input}} =
E_{\text{token}} +
E_{\text{position}} +
E_{\text{segment}}
\]

Hasilnya diteruskan ke lapisan Transformer Encoder.

Representasi token keluaran BERT bersifat kontekstual. Karena itu, representasi kata yang sama dapat berbeda ketika digunakan dalam kalimat berbeda.

---

## 9. Pre-training BERT

BERT dilatih terlebih dahulu pada korpus teks besar melalui objective pembelajaran bahasa. BERT asli menggunakan dua objective utama berikut.

### 9.1 Masked Language Modeling (MLM)

Sebagian token dipilih untuk proses masking. Model dilatih untuk memprediksi token asli berdasarkan konteks di sekitarnya.

Contoh konseptual:

```text
Input:
Saya membutuhkan [MASK] karena rumah terendam banjir.

Target:
bantuan
```

Contoh ini hanya ilustrasi. Dalam prosedur pelatihan aslinya, pemilihan token dan cara menggantinya mengikuti aturan tertentu; tidak semua token terpilih selalu diganti secara literal dengan `[MASK]`.

### 9.2 Next Sentence Prediction (NSP)

Dalam BERT asli, model juga dilatih untuk memprediksi apakah dua segmen teks disusun sebagai pasangan berurutan yang benar menurut data pelatihan.

NSP merupakan bagian dari pelatihan BERT asli, tetapi **tidak semua model turunan BERT memakai objective yang sama**. Model bahasa yang lebih baru dapat menggunakan strategi pre-training berbeda.

### 9.3 Pre-training vs fine-tuning

| Aspek | Pre-training | Fine-tuning |
|---|---|---|
| Tujuan | Mempelajari representasi bahasa umum | Menyesuaikan model untuk tugas tertentu |
| Data | Korpus teks besar | Dataset tugas spesifik dan berlabel untuk supervised task |
| Label tugas | Tidak harus berupa label klasifikasi | Umumnya diperlukan untuk klasifikasi supervised |
| Hasil | Model bahasa pralatih | Model yang disesuaikan untuk tugas |

---

## 10. Fine-tuning BERT

Fine-tuning adalah proses melatih model pralatih lebih lanjut pada dataset tugas tertentu.

Untuk klasifikasi teks, arsitektur umumnya:

```text
Teks
  ↓
Tokenizer
  ↓
BERT Encoder
  ↓
Representasi [CLS] (pada pola klasifikasi umum)
  ↓
Classification Head
  ↓
Logits untuk tiap kelas
  ↓
Prediksi kelas
```

Classification head biasanya berupa lapisan linear atau komponen kecil lain di atas representasi BERT. Selama fine-tuning, parameter BERT dan head umumnya diperbarui bersama, kecuali jika sebagian parameter sengaja dibekukan.

### Contoh tugas

Label:

- `0`: Non-Emergency
- `1`: Emergency

Input:

```text
"Tolong bantu kami, rumah sudah terendam banjir."
```

Model menghasilkan skor untuk kedua kelas, lalu skor tersebut digunakan untuk menentukan prediksi.

Jangan menyamakan skor softmax dengan probabilitas yang sudah terkalibrasi sempurna. Nilai confidence tetap perlu ditafsirkan dengan hati-hati.

---

## 11. Praktik 1: Tokenizer dengan Hugging Face

### 11.1 Instalasi

Jalankan di terminal atau notebook:

```bash
pip install transformers torch
```

Jika menggunakan environment baru, pastikan instalasi PyTorch sesuai sistem dan dukungan perangkat keras Anda.

### 11.2 Kode

Contoh berikut menggunakan checkpoint BERT berbahasa Inggris untuk memperlihatkan mekanisme tokenizer. Ini **bukan pilihan model yang disarankan untuk teks bahasa Indonesia**.

```python
from transformers import AutoTokenizer

checkpoint = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

text = "Flood victims need emergency assistance."

encoded = tokenizer(
    text,
    padding=False,
    truncation=True,
    return_tensors=None
)

print("Tokens:", tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
print("Input IDs:", encoded["input_ids"])
print("Attention mask:", encoded["attention_mask"])
```

### 11.3 Bentuk output

Output token dan angka ID bergantung pada checkpoint dan versi tokenizer. Secara umum, bentuknya seperti:

```text
Tokens: ['[CLS]', 'flood', 'victims', 'need', 'emergency',
         'assistance', '.', '[SEP]']

Input IDs: [101, ..., ..., ..., ..., ..., 1012, 102]
Attention mask: [1, 1, 1, 1, 1, 1, 1, 1]
```

Angka `...` sengaja tidak diisi karena ID token harus berasal dari hasil eksekusi tokenizer yang sebenarnya. Jangan menyalin ID ilustratif sebagai nilai tetap.

### 11.4 Menggunakan padding dan truncation

```python
texts = [
    "Flood victims need assistance.",
    "Please send help to the affected village."
]

batch = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=16,
    return_tensors="pt"
)

print("Input IDs shape:", batch["input_ids"].shape)
print("Attention mask shape:", batch["attention_mask"].shape)
print("Input IDs:")
print(batch["input_ids"])
print("Attention mask:")
print(batch["attention_mask"])
```

Bentuk output yang diharapkan:

```text
Input IDs shape: torch.Size([2,  ...])
Attention mask shape: torch.Size([2,  ...])
```

Dimensi kedua mengikuti panjang batch setelah padding, dengan batas maksimum 16 token. Nilai tensor yang tepat bergantung pada tokenizer.

---

## 12. Praktik 2: Mengambil Representasi BERT

Bagian ini menunjukkan cara memperoleh hidden states dari model BERT tanpa classification head.

### Kode

```python
import torch
from transformers import AutoTokenizer, AutoModel

checkpoint = "google-bert/bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)
model.eval()

text = "Flood victims need emergency assistance."
inputs = tokenizer(text, return_tensors="pt", truncation=True)

with torch.no_grad():
    outputs = model(**inputs)

print("Input IDs shape:", inputs["input_ids"].shape)
print("Last hidden state shape:", outputs.last_hidden_state.shape)
```

### Memahami output

Bentuk umum `last_hidden_state` adalah:

```text
(batch_size, sequence_length, hidden_size)
```

Untuk BERT-base, `hidden_size` biasanya `768`. Jadi untuk satu teks, output memiliki bentuk:

```text
(1, jumlah_token, 768)
```

Jumlah token ditentukan oleh tokenizer, termasuk token khusus. Jika jumlah token adalah 8, bentuknya menjadi:

```text
torch.Size([1, 8, 768])
```

Angka `8` di sini merupakan contoh bentuk output, bukan jaminan panjang untuk setiap input.

### Mengambil representasi token `[CLS]`

```python
cls_vector = outputs.last_hidden_state[:, 0, :]

print("CLS vector shape:", cls_vector.shape)
```

Output yang diharapkan:

```text
CLS vector shape: torch.Size([1, 768])
```

Representasi `[CLS]` lazim digunakan pada pola klasifikasi BERT, tetapi cara pooling terbaik dapat bergantung pada model dan tugas. Jangan menganggap vector `[CLS]` sebagai embedding universal yang selalu optimal untuk pencarian kemiripan.

---

## 13. Praktik 3: Klasifikasi Teks dengan Pipeline

Hugging Face menyediakan `pipeline` agar kita dapat menjalankan tugas menggunakan model yang sudah tersedia.

> Contoh ini menggunakan model analisis sentimen bahasa Inggris untuk demonstrasi API. Hasilnya tidak boleh dipakai sebagai hasil penelitian bahasa Indonesia.

### Kode

```python
from transformers import pipeline

classifier = pipeline(
    task="sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

result = classifier("The emergency team helped the flood victims.")

print(result)
```

### Output

Output akan berupa daftar berisi label dan skor, misalnya dengan bentuk:

```text
[{'label': 'POSITIVE', 'score': 0.99...}]
```

Skor tepatnya dapat berbeda tergantung versi model, library, dan lingkungan eksekusi.

### Catatan penting

- Pipeline ini mempermudah demonstrasi inferensi.
- Model yang digunakan sudah dilatih untuk sentimen bahasa Inggris.
- Pipeline tersebut **bukan contoh fine-tuning** dan bukan model untuk mendeteksi permintaan bantuan.
- Untuk penelitian, gunakan dataset yang sesuai dan latih/evaluasi model dengan metodologi yang benar.

---

## 14. BERT untuk Klasifikasi Permintaan Bantuan

Misalkan tujuan penelitian adalah mengklasifikasikan postingan media sosial terkait bencana.

### 14.1 Definisi label

Contoh label awal:

| Label | Arti | Contoh ilustratif |
|---|---|---|
| `0` — Non-Emergency | Tidak menunjukkan kebutuhan bantuan darurat secara langsung | "Pemerintah mengumumkan status siaga banjir." |
| `1` — Emergency | Mengandung permintaan bantuan atau indikasi kebutuhan darurat yang jelas | "Tolong evakuasi keluarga kami, air sudah masuk rumah." |

Contoh ini belum cukup untuk menjadi pedoman anotasi. Dalam penelitian nyata, buat aturan yang rinci untuk kasus ambigu, laporan tidak langsung, berita, informasi lokasi, dan permintaan bantuan yang sudah tertangani.

### 14.2 Alur eksperimen

```text
Dataset berlabel
      ↓
Pemeriksaan kualitas label
      ↓
Train / Validation / Test split
      ↓
Tokenizer BERT yang sesuai
      ↓
Fine-tuning model
      ↓
Evaluasi pada test set
      ↓
Analisis kesalahan
```

### 14.3 Mengapa perlu baseline?

Bandingkan BERT dengan model sederhana seperti:

```text
TF-IDF + Logistic Regression
TF-IDF + Linear SVM
BERT / IndoBERT fine-tuned
```

Baseline membantu menjawab apakah model Transformer benar-benar memberikan peningkatan pada dataset penelitian.

### 14.4 Mencegah data leakage

- Pisahkan data train, validation, dan test sebelum mengambil keputusan berdasarkan performa.
- Jangan menggunakan test set untuk memilih hyperparameter.
- Periksa tweet duplikat atau hampir duplikat agar salinan pesan tidak bocor ke beberapa split.
- Jika data berasal dari peristiwa yang sama, pertimbangkan split berdasarkan waktu atau peristiwa untuk menguji generalisasi yang lebih realistis.

### 14.5 Metrik

Jangan hanya melaporkan accuracy. Pertimbangkan:

- Precision per kelas
- Recall kelas Emergency
- F1-score
- Macro F1
- Confusion matrix

Recall kelas Emergency penting karena false negative berarti pesan yang membutuhkan bantuan terlewat. Namun, precision juga penting agar sistem tidak membanjiri petugas dengan terlalu banyak alarm palsu.

---

## 15. BERT vs TF-IDF dan Word2Vec

| Aspek | TF-IDF | Word2Vec | BERT |
|---|---|---|---|
| Representasi | Bobot kata/dokumen | Vektor kata yang umumnya statis | Representasi token kontekstual |
| Mempertimbangkan konteks kalimat | Terbatas | Tidak secara dinamis pada pemakaian standar | Ya |
| Urutan token | Tidak dipertahankan langsung oleh fitur BoW/TF-IDF | Tidak ditangani oleh vektor kata saja | Ditangani oleh Transformer |
| Kebutuhan komputasi | Relatif rendah | Relatif rendah saat inferensi embedding | Umumnya lebih tinggi |
| Kebutuhan data untuk mulai | Dapat bekerja pada dataset kecil | Perlu data untuk melatih embedding, atau gunakan embedding pralatih | Dapat memanfaatkan model pralatih dan fine-tuning |
| Peran dalam penelitian | Baseline kuat dan mudah dijelaskan | Fitur embedding untuk model lain | Kandidat model kontekstual |

Tidak ada satu pendekatan yang selalu unggul. Pilihan harus berdasarkan tujuan, data, biaya komputasi, dan hasil evaluasi.

---

## 16. Kelebihan dan Keterbatasan BERT

### Kelebihan

- Menghasilkan representasi yang mempertimbangkan konteks.
- Dapat diadaptasi untuk banyak tugas NLP.
- Memanfaatkan pengetahuan dari pre-training.
- Fine-tuning sering lebih efisien daripada melatih model bahasa dari nol.

### Keterbatasan

- Lebih berat secara komputasi dibandingkan TF-IDF.
- Panjang input terbatas sesuai konfigurasi model.
- Dapat menghasilkan prediksi yang bias atau salah.
- Memerlukan evaluasi dan penanganan data yang teliti.
- Model bahasa Inggris tidak otomatis cocok untuk bahasa Indonesia.
- Confidence score tidak selalu mencerminkan tingkat kepastian sebenarnya.

---

## 17. Kesalahan Umum

1. **Menganggap BERT sama dengan GPT.** BERT klasik adalah model encoder, sedangkan GPT klasik menggunakan pendekatan decoder autoregresif.
2. **Menghapus semua stopword sebelum BERT tanpa alasan.** Penghapusan kata dapat mengubah konteks. Mulailah dari teks yang dibersihkan secara wajar dan uji keputusan preprocessing.
3. **Menganggap ID token selalu sama.** ID token bergantung pada tokenizer dan kosakata checkpoint.
4. **Menganggap attention mask sebagai label.** Mask umumnya membedakan token nyata dari padding; mask bukan label klasifikasi.
5. **Menggunakan model bahasa Inggris untuk kesimpulan bahasa Indonesia.** Pilih checkpoint yang sesuai dengan bahasa dan tugas.
6. **Mengklaim model akurat hanya dari training set.** Laporkan evaluasi pada data yang tidak dipakai untuk melatih model.
7. **Menganggap BERT pasti lebih unggul dari TF-IDF + SVM.** Buktikan melalui eksperimen yang adil.

---

## 18. Latihan

### Latihan 1 — Konsep

Jawab dengan kalimat sendiri:

1. Apa kepanjangan BERT?
2. Mengapa BERT disebut bidirectional?
3. Apa peran `[CLS]` dan `[SEP]`?
4. Apa perbedaan pre-training dan fine-tuning?
5. Apa perbedaan `input_ids` dan `attention_mask`?
6. Mengapa BERT menggunakan subword tokenization?
7. Mengapa BERT perlu dibandingkan dengan baseline?

### Latihan 2 — Tokenizer

Ubah kode Praktik 1 agar:

- menerima tiga kalimat;
- menampilkan token;
- menampilkan jumlah token;
- menggunakan `padding=True`;
- menampilkan bentuk `input_ids`.

### Latihan 3 — Representasi

Jalankan Praktik 2 dan jawab:

1. Apa arti setiap dimensi pada `last_hidden_state`?
2. Mengapa panjang sequence dapat berubah?
3. Apa bentuk vector `[CLS]` untuk satu teks?
4. Apakah mengambil vector `[CLS]` sama dengan melakukan fine-tuning klasifikasi?

### Latihan 4 — Desain penelitian

Buat rancangan eksperimen untuk klasifikasi `Emergency` dan `Non-Emergency`. Jelaskan:

- definisi label;
- cara membagi dataset;
- baseline;
- model BERT yang sesuai;
- metrik;
- cara menghindari data leakage.

---

## 19. Mini Project

### Judul

**Eksplorasi Tokenizer dan Representasi BERT untuk Teks Bencana**

### Tujuan

Membuat notebook yang memperlihatkan bagaimana teks diubah menjadi token dan direpresentasikan oleh BERT.

### Tugas

1. Siapkan 10–20 kalimat contoh tentang bencana.
2. Gunakan tokenizer dan model yang sama.
3. Tampilkan token, `input_ids`, dan `attention_mask`.
4. Ambil `last_hidden_state` dan vector `[CLS]`.
5. Bandingkan bentuk tensor untuk kalimat pendek dan panjang.
6. Tulis analisis singkat mengenai apa yang dilakukan model.
7. Jangan mengklaim bahwa contoh kecil ini membuktikan performa klasifikasi.

### Struktur folder

```text
bert-mini-project/
├── README.md
├── requirements.txt
├── notebooks/
│   └── bert_exploration.ipynb
└── reports/
    └── observations.md
```

Contoh `requirements.txt`:

```text
transformers
torch
```

---

## 20. Checklist Kompetensi

Gunakan checklist berikut sebelum melanjutkan ke IndoBERT.

- [ ] Saya dapat menjelaskan BERT dan Transformer Encoder.
- [ ] Saya memahami konsep bidirectional context.
- [ ] Saya mengenali token khusus BERT.
- [ ] Saya dapat menjelaskan `input_ids`, `attention_mask`, dan padding.
- [ ] Saya memahami token, subword, dan embedding.
- [ ] Saya dapat menggunakan `AutoTokenizer`.
- [ ] Saya dapat memuat model dengan `AutoModel`.
- [ ] Saya memahami bentuk `last_hidden_state`.
- [ ] Saya memahami pre-training dan fine-tuning.
- [ ] Saya memahami mengapa model harus dievaluasi pada test set.
- [ ] Saya memahami alasan memakai baseline TF-IDF + SVM.

Jika sebagian besar checklist sudah terpenuhi, lanjutkan ke materi **IndoBERT**, kemudian praktik **fine-tuning IndoBERT untuk klasifikasi teks bahasa Indonesia**.

---

## Referensi untuk Belajar Lebih Lanjut

- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT.
- Dokumentasi Hugging Face Transformers: <https://huggingface.co/docs/transformers>
- Dokumentasi Auto Classes: <https://huggingface.co/docs/transformers/model_doc/auto>
- Dokumentasi BERT: <https://huggingface.co/docs/transformers/model_doc/bert>

> **Catatan akhir:** Materi ini menggunakan contoh checkpoint BERT berbahasa Inggris untuk menjelaskan API dan konsep. Untuk penelitian teks bahasa Indonesia, gunakan model yang sesuai—misalnya IndoBERT—dan lakukan evaluasi empiris pada dataset Anda sendiri.
