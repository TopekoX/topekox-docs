# Word Embedding

## 1. Tujuan Pembelajaran

Setelah mempelajari materi ini, Anda diharapkan mampu:

1. Menjelaskan pengertian dan fungsi Word Embedding.
2. Memahami perbedaan One-Hot Encoding, Bag of Words, TF-IDF, dan Word Embedding.
3. Memahami konsep vektor, dimensi, dan kedekatan semantik kata.
4. Memahami metode Word2Vec, CBOW, dan Skip-Gram.
5. Membuat Word Embedding menggunakan Python.
6. Mengukur kemiripan kata menggunakan Cosine Similarity.
7. Memahami keterbatasan Word Embedding klasik dan hubungannya dengan BERT/IndoBERT.

## 2. Apa Itu Word Embedding?

**Word Embedding adalah teknik untuk merepresentasikan kata sebagai vektor numerik berdimensi tertentu sehingga pola hubungan antarkata dapat dipelajari oleh komputer.**

Komputer tidak secara langsung memahami makna kata seperti manusia. Misalnya, teks berikut:

```text
banjir
bantuan
korban
sekolah
```

dapat direpresentasikan sebagai vektor angka.

| Kata | Representasi vektor (ilustrasi) |
|---|---|
| banjir | `[0.21, -0.43, 0.82, ...]` |
| bantuan | `[0.35, 0.18, 0.71, ...]` |
| korban | `[0.29, -0.12, 0.65, ...]` |
| sekolah | `[-0.42, 0.76, -0.18, ...]` |

Angka tersebut hanya ilustrasi, bukan hasil pelatihan model nyata.

Setiap kata direpresentasikan oleh vektor dengan jumlah dimensi tertentu. Contohnya, embedding 100 dimensi berarti satu kata direpresentasikan oleh 100 angka.

### 2.1 Alur Word Embedding

```text
Teks
"Korban banjir membutuhkan bantuan"
        |
        v
Tokenisasi
["Korban", "banjir", "membutuhkan", "bantuan"]
        |
        v
Embedding
Setiap token dipetakan ke vektor numerik
        |
        v
Representasi numerik
[0.21, -0.43, ...]
```

Vektor tersebut kemudian dapat digunakan oleh model Machine Learning atau Deep Learning untuk mempelajari pola dari teks.

## 3. Mengapa Word Embedding Dibutuhkan?

Sebelum mempelajari Word Embedding, pahami keterbatasan representasi teks klasik.

Contoh corpus:

```text
Dokumen 1: korban membutuhkan bantuan
Dokumen 2: korban memerlukan pertolongan
Dokumen 3: siswa membaca buku
```

Secara makna, kata *bantuan* dan *pertolongan* memiliki hubungan yang dekat. Namun, Bag of Words dan TF-IDF tidak secara langsung mengetahui bahwa kedua kata tersebut mirip makna.

Word Embedding dapat mempelajari hubungan tersebut dari pola kemunculan kata dalam data pelatihan.

**Word Embedding tidak otomatis memahami makna seperti manusia.** Kualitas representasinya bergantung pada data, metode, dan proses pelatihan.

## 4. Perbedaan Representasi Teks

| Metode | Karakteristik | Keterbatasan utama |
|---|---|---|
| One-Hot Encoding | Satu posisi bernilai 1 untuk setiap kata | Dimensi besar dan tidak merepresentasikan kemiripan makna |
| Bag of Words | Menghitung frekuensi kata | Tidak memahami urutan kata dan hubungan semantik secara langsung |
| TF-IDF | Memberi bobot berdasarkan frekuensi dan kelangkaan kata | Tidak secara langsung mempelajari kemiripan semantik |
| Word2Vec | Mempelajari vektor kata dari konteks | Umumnya satu kata memiliki satu vektor, meskipun konteksnya berbeda |
| Contextual Embedding | Representasi dipengaruhi konteks kalimat | Memerlukan model kontekstual yang lebih kompleks |

Contoh perbedaan:

- **TF-IDF:** kata *bantuan* direpresentasikan berdasarkan bobotnya dalam dokumen.
- **Word2Vec:** kata *bantuan* direpresentasikan berdasarkan pola konteks yang dipelajari selama pelatihan.
- **IndoBERT:** representasi token dapat berubah berdasarkan konteks kalimat tempat token digunakan.

## 5. Memahami Vektor dan Dimensi

Vektor adalah kumpulan angka yang merepresentasikan sebuah objek. Dalam Word Embedding, objek tersebut berupa kata atau token.

Contoh:

\[
v_{\text{banjir}} = [0.2,\ 0.7,\ -0.4]
\]

Vektor tersebut mempunyai tiga dimensi.

Setiap dimensi merupakan nilai yang dipelajari model. Pada embedding modern, dimensi-dimensi tersebut biasanya tidak memiliki arti sederhana yang dapat ditafsirkan secara terpisah.

### 5.1 Praktik Python: Membuat Vektor Sederhana

```python
import numpy as np

embedding = {
    "banjir": np.array([0.2, 0.7, -0.4]),
    "bantuan": np.array([0.3, 0.6, -0.2]),
    "sekolah": np.array([-0.5, 0.1, 0.8])
}

for word, vector in embedding.items():
    print(f"Kata: {word}")
    print(f"Vektor: {vector}")
    print(f"Dimensi: {len(vector)}")
    print()
```

Output:

```text
Kata: banjir
Vektor: [ 0.2  0.7 -0.4]
Dimensi: 3

Kata: bantuan
Vektor: [ 0.3  0.6 -0.2]
Dimensi: 3

Kata: sekolah
Vektor: [-0.5  0.1  0.8]
Dimensi: 3
```

Vektor di atas dibuat secara manual untuk latihan. Dalam model Word2Vec, nilai vektor dipelajari dari data pelatihan.

## 6. Bagaimana Word Embedding Dipelajari?

Salah satu pendekatan yang populer adalah mempelajari hubungan kata melalui konteks kemunculannya.

Contoh:

```text
Korban banjir membutuhkan bantuan
Warga banjir membutuhkan bantuan
Korban gempa membutuhkan bantuan
```

Kata *korban*, *banjir*, *gempa*, dan *bantuan* muncul dalam pola konteks tertentu. Model belajar mengenali pola statistik tersebut selama pelatihan.

### 6.1 Konsep Utama

- **Corpus:** kumpulan teks untuk melatih model.
- **Vocabulary:** kumpulan token yang dikenali model.
- **Context window:** sejumlah token di sekitar kata target yang digunakan untuk belajar.
- **Embedding dimension:** jumlah angka yang digunakan untuk merepresentasikan setiap kata.
- **Training:** proses memperbarui parameter model agar tugas prediksi kata menjadi lebih baik.

Contoh context window berukuran 2:

```text
Kalimat:
Korban banjir membutuhkan bantuan segera

Target:
membutuhkan

Konteks:
Korban, banjir, bantuan, segera
```

Dalam contoh tersebut, model menggunakan dua kata di sebelah kiri dan dua kata di sebelah kanan sebagai konteks.

## 7. Word2Vec

Word2Vec adalah metode pembelajaran representasi kata yang diperkenalkan oleh Tomas Mikolov dan rekan-rekannya.

Word2Vec memiliki dua arsitektur utama.

### 7.1 CBOW (Continuous Bag of Words)

Model menggunakan kata-kata di sekitar suatu kata untuk memprediksi kata target.

Contoh:

```text
Konteks: korban banjir ___ bantuan
Target: membutuhkan
```

### 7.2 Skip-Gram

Model menggunakan kata target untuk memprediksi kata-kata di sekitarnya.

Contoh:

```text
Target: membutuhkan
Konteks yang diprediksi: banjir, bantuan
```

Perbedaan keduanya terletak pada arah prediksi, bukan pada tujuan akhirnya. Keduanya berusaha mempelajari representasi kata dari konteks.

### 7.3 Perbandingan CBOW dan Skip-Gram

| Aspek | CBOW | Skip-Gram |
|---|---|---|
| Input | Kata konteks | Kata target |
| Prediksi | Kata target | Kata konteks |
| Karakteristik umum | Sering efisien untuk data besar | Dapat bermanfaat untuk kata yang jarang muncul |
| Tujuan | Mempelajari embedding kata | Mempelajari embedding kata |

Karakteristik tersebut bukan jaminan performa. Hasil aktual bergantung pada ukuran corpus, parameter, dan distribusi kata.

## 8. Implementasi Word2Vec dengan Python

Pada bagian ini, kita membangun Word Embedding menggunakan pustaka Gensim.

### 8.1 Instalasi

Jalankan pada terminal atau Jupyter Notebook:

```bash
pip install gensim numpy
```

### 8.2 Menyiapkan Corpus

Untuk Word2Vec, teks umumnya ditokenisasi menjadi daftar token.

```python
sentences = [
    ["korban", "banjir", "membutuhkan", "bantuan"],
    ["warga", "terdampak", "banjir", "membutuhkan", "bantuan"],
    ["korban", "gempa", "membutuhkan", "pertolongan"],
    ["tim", "relawan", "memberikan", "bantuan"],
    ["warga", "mencari", "tempat", "pengungsian"],
    ["siswa", "belajar", "di", "sekolah"],
    ["guru", "mengajar", "di", "sekolah"]
]

for sentence in sentences:
    print(sentence)
```

Output:

```text
['korban', 'banjir', 'membutuhkan', 'bantuan']
['warga', 'terdampak', 'banjir', 'membutuhkan', 'bantuan']
['korban', 'gempa', 'membutuhkan', 'pertolongan']
['tim', 'relawan', 'memberikan', 'bantuan']
['warga', 'mencari', 'tempat', 'pengungsian']
['siswa', 'belajar', 'di', 'sekolah']
['guru', 'mengajar', 'di', 'sekolah']
```

Corpus ini hanya untuk demonstrasi. Terlalu kecil untuk menghasilkan embedding yang andal untuk penelitian.

### 8.3 Melatih Model Word2Vec

```python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=2,
    min_count=1,
    workers=1,
    sg=1,
    seed=42,
    epochs=100
)

print("Ukuran vocabulary:", len(model.wv))
print("Dimensi embedding:", model.wv.vector_size)
```

Output:

```text
Ukuran vocabulary: 20
Dimensi embedding: 50
```

Jumlah vocabulary mengikuti kata unik yang ada dalam corpus. Dimensi embedding ditetapkan menjadi 50 melalui `vector_size=50`.

Arti parameter:

| Parameter | Penjelasan |
|---|---|
| `sentences` | Corpus yang telah ditokenisasi |
| `vector_size=50` | Setiap kata memiliki vektor 50 dimensi |
| `window=2` | Ukuran jendela konteks |
| `min_count=1` | Kata yang muncul minimal sekali dipertahankan |
| `sg=1` | Menggunakan Skip-Gram |
| `workers=1` | Menggunakan satu worker |
| `seed=42` | Membantu reproduksibilitas |
| `epochs=100` | Jumlah putaran pelatihan |

Untuk menggunakan CBOW, ubah `sg=1` menjadi `sg=0`.

### 8.4 Mengambil Embedding Kata

```python
vector = model.wv["banjir"]

print("Vektor banjir:")
print(vector)

print("Dimensi:", len(vector))
```

Output menampilkan 50 angka yang dipelajari selama training. Nilai persisnya dapat bergantung pada lingkungan dan versi pustaka.

Contoh bentuk output:

```text
Vektor banjir:
[ 0.012 ... -0.034 ...]

Dimensi: 50
```

Tanda `...` menunjukkan bahwa sebagian nilai dihilangkan dari ilustrasi, bukan output literal Python.

### 8.5 Memeriksa Vocabulary

```python
print(model.wv.key_to_index.keys())
```

Perintah ini menampilkan kata-kata yang dikenali model beserta indeksnya dalam vocabulary.

Jika kata tidak ada dalam vocabulary, pemanggilan seperti `model.wv["tsunami"]` akan menghasilkan `KeyError`. Untuk penggunaan nyata, model harus dilatih dengan corpus yang cukup representatif atau menggunakan pendekatan yang mampu menangani kata yang belum pernah dilihat.

## 9. Mengukur Kemiripan Kata dengan Cosine Similarity

Salah satu metode umum untuk membandingkan dua embedding adalah **cosine similarity**.

\[
\operatorname{cos}(A,B) =
\frac{A \cdot B}{\|A\|\|B\|}
\]

Nilainya secara matematis berada di antara -1 dan 1 untuk vektor nonnol.

- Mendekati 1: arah vektor serupa.
- Mendekati 0: vektor relatif tegak lurus.
- Mendekati -1: arah vektor berlawanan.

Nilai ini bukan ukuran kebenaran makna kata secara mutlak.

### 9.1 Praktik Python

```python
from sklearn.metrics.pairwise import cosine_similarity

v_banjir = model.wv["banjir"].reshape(1, -1)
v_bantuan = model.wv["bantuan"].reshape(1, -1)
v_sekolah = model.wv["sekolah"].reshape(1, -1)

sim_banjir_bantuan = cosine_similarity(
    v_banjir, v_bantuan
)[0][0]

sim_banjir_sekolah = cosine_similarity(
    v_banjir, v_sekolah
)[0][0]

print("Banjir - Bantuan:", sim_banjir_bantuan)
print("Banjir - Sekolah:", sim_banjir_sekolah)
```

Hasil numeriknya bergantung pada embedding yang dipelajari. Jangan mengasumsikan bahwa kemiripan `banjir` dan `bantuan` pasti lebih tinggi daripada `banjir` dan `sekolah`, terutama karena corpus demonstrasi ini sangat kecil.

### 9.2 Mencari Kata yang Mirip

Gensim menyediakan fungsi untuk mencari kata terdekat:

```python
similar_words = model.wv.most_similar(
    "banjir",
    topn=3
)

for word, score in similar_words:
    print(f"{word}: {score:.4f}")
```

Output berbentuk:

```text
kata_1: skor_kemiripan
kata_2: skor_kemiripan
kata_3: skor_kemiripan
```

Kata dan skornya berasal dari model yang benar-benar dilatih. Dengan corpus kecil, hasilnya belum layak ditafsirkan sebagai pengetahuan bahasa yang andal.

## 10. Keterbatasan Word2Vec

### 10.1 Satu Kata, Satu Embedding

Word2Vec klasik memberikan satu vektor untuk setiap kata dalam vocabulary.

Contoh:

```text
Banjir melanda wilayah tersebut.
Informasi itu membanjiri media sosial.
```

Kata *membanjiri* digunakan secara literal pada konteks pertama dan metaforis pada konteks kedua. Word2Vec klasik tidak menghasilkan embedding berbeda berdasarkan konteks kalimat tersebut.

### 10.2 Kata di Luar Vocabulary

Word2Vec umumnya tidak dapat menghasilkan embedding untuk kata yang tidak ada dalam vocabulary ketika training.

Hal ini menjadi tantangan untuk media sosial, karena banyak kata informal, singkatan, salah ketik, dan variasi ejaan.

### 10.3 Ketergantungan pada Corpus

Embedding yang dilatih dengan corpus berita mungkin berbeda dari embedding yang dilatih dengan percakapan media sosial.

Untuk penelitian, pemilihan data pelatihan dan evaluasi sangat penting.

## 11. Dari Word2Vec ke Contextual Embedding

Perkembangan representasi teks dapat diringkas sebagai berikut:

```text
TF-IDF
Bobot kata berdasarkan dokumen
        |
        v
Word2Vec / FastText
Vektor kata yang dipelajari dari konteks
        |
        v
Transformer / BERT
Representasi yang dipengaruhi konteks kalimat
        |
        v
IndoBERT
Representasi kontekstual untuk bahasa Indonesia
```

Ini merupakan gambaran perkembangan pendekatan, bukan berarti TF-IDF harus selalu diganti Word2Vec atau Word2Vec harus selalu digunakan sebelum BERT. Untuk penelitian klasifikasi teks, TF-IDF dapat tetap menjadi baseline yang kuat dan berguna.

## 12. Mengenal FastText

FastText merupakan pendekatan embedding yang dikembangkan oleh Facebook AI Research. Salah satu kelebihannya adalah penggunaan informasi subkata (*subword*).

Contoh kata:

```text
membantu
bantuan
membutuhkan
```

FastText mempelajari representasi kata dengan memanfaatkan karakter n-gram. Hal ini dapat membantu menangani variasi morfologi dan kata yang tidak pernah muncul persis dalam vocabulary, meskipun bukan jaminan bahwa semua typo atau slang dapat ditangani dengan benar.

FastText layak dipelajari setelah Word2Vec, khususnya jika Anda bekerja dengan teks bahasa Indonesia yang mempunyai banyak imbuhan dan variasi penulisan.

## 13. Hubungan Word Embedding dengan IndoBERT

IndoBERT juga menggunakan embedding, tetapi arsitekturnya lebih kompleks daripada Word2Vec.

Pada IndoBERT, teks dipecah menjadi token menggunakan tokenizer model. Token kemudian direpresentasikan melalui embedding dan diproses oleh lapisan Transformer sehingga representasinya dipengaruhi konteks.

Contoh:

```text
Kalimat A:
Korban membutuhkan bantuan.

Kalimat B:
Bantuan tersebut diberikan kepada korban.
```

Pada model kontekstual, representasi token dipengaruhi oleh kata-kata di sekitarnya. Tokenisasi IndoBERT juga menggunakan subword, sehingga satu kata dapat terdiri atas beberapa token.

### 13.1 Mengambil Representasi IndoBERT

```python
import torch
from transformers import AutoTokenizer, AutoModel

model_name = "indobenchmark/indobert-base-p1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)

text = "Korban banjir membutuhkan bantuan."

inputs = tokenizer(
    text,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = bert_model(**inputs)

print("Input IDs:", inputs["input_ids"])
print("Hidden state shape:", outputs.last_hidden_state.shape)
```

Output `hidden state shape` berbentuk:

```text
torch.Size([1, jumlah_token, hidden_size])
```

Untuk model IndoBERT base yang umum digunakan, `hidden_size` adalah 768. Jumlah token bergantung pada hasil tokenisasi, termasuk token khusus.

`last_hidden_state` berisi representasi kontekstual setiap token. Ini berbeda dari vektor statis Word2Vec.

Kode ini memerlukan akses untuk mengunduh model pertama kali dan lingkungan dengan pustaka `torch` serta `transformers` yang kompatibel.

Instalasi pustaka jika diperlukan:

```bash
pip install torch transformers
```

## 14. Penerapan untuk Penelitian Deteksi Permintaan Bantuan Darurat

Dalam penelitian, Word Embedding dapat digunakan untuk memahami pendekatan representasi teks sebelum membangun model IndoBERT.

Contoh alur penelitian:

```text
Tweet Bahasa Indonesia
        |
        v
Pembersihan teks yang sesuai
        |
        v
Dataset berlabel
        |
        v
Baseline TF-IDF + SVM
        |
        v
Eksperimen Word2Vec / FastText
        |
        v
Fine-tuning IndoBERT
        |
        v
Evaluasi dan perbandingan
```

Anda tidak wajib memakai Word2Vec sebagai model utama. Word2Vec lebih tepat dipelajari untuk memahami evolusi representasi kata dan menjadi dasar konseptual sebelum contextual embedding.

Untuk penelitian yang baik, bandingkan model dengan pembagian data yang konsisten, metrik yang sesuai, dan pencegahan data leakage. Pada deteksi permintaan bantuan darurat, perhatikan khususnya **recall kelas Emergency** dan **F1-score**.

## 15. Latihan Mandiri

Kerjakan latihan berikut secara berurutan.

### Latihan 1 — Vektor

Buat tiga vektor kata secara manual dan periksa dimensinya.

### Latihan 2 — Corpus

Buat minimal 20 kalimat berbahasa Indonesia yang berkaitan dengan bencana.

### Latihan 3 — Word2Vec

Latih model CBOW dan Skip-Gram dengan corpus yang sama.

### Latihan 4 — Similarity

Bandingkan cosine similarity beberapa pasangan kata.

### Latihan 5 — Analisis Parameter

Ubah ukuran `window` dan dimensi embedding, lalu amati perubahan hasil.

### Latihan 6 — IndoBERT

Ambil `last_hidden_state` IndoBERT dan periksa dimensi output-nya.

## 16. Ringkasan

Hal-hal terpenting yang perlu diingat:

- Word Embedding merepresentasikan kata sebagai vektor numerik.
- Word2Vec mempelajari vektor kata melalui konteks, dengan arsitektur CBOW dan Skip-Gram.
- Cosine Similarity dapat digunakan untuk membandingkan arah vektor.
- Word2Vec klasik menggunakan embedding statis, sedangkan BERT menghasilkan representasi kontekstual.
- FastText memanfaatkan informasi subkata untuk membantu merepresentasikan variasi kata.
- Dalam penelitian, TF-IDF + SVM tetap berguna sebagai baseline untuk dibandingkan dengan IndoBERT.

**Tahap berikutnya dalam roadmap NLP:** pelajari dasar Deep Learning dan PyTorch, khususnya tensor, embedding layer, loss function, optimizer, dan training loop. Setelah itu, lanjutkan ke RNN/LSTM secara konseptual sebelum mendalami Attention dan Transformer.
