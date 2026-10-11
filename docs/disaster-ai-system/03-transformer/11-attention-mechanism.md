# Attention Mechanism untuk NLP

> **Posisi dalam roadmap:** RNN → LSTM → GRU → **Attention Mechanism** → Transformer → BERT → IndoBERT  
> **Tujuan:** memahami bagaimana model memberi bobot pada bagian input yang relevan, lalu membangun fondasi untuk memahami Self-Attention dan Transformer.

## Tujuan Pembelajaran

Setelah mempelajari materi ini, Anda diharapkan mampu:

1. Menjelaskan masalah yang ingin diselesaikan oleh Attention Mechanism.
2. Memahami konsep **Query (Q), Key (K), dan Value (V)**.
3. Menghitung attention score, attention weight, dan context vector secara sederhana.
4. Membedakan attention pada encoder-decoder dengan self-attention.
5. Memahami scaled dot-product attention dan multi-head attention secara konseptual.
6. Menjalankan contoh Attention sederhana menggunakan Python.
7. Menjelaskan hubungan Attention dengan Transformer dan IndoBERT.

---

## 1. Apa Itu Attention Mechanism?

**Attention Mechanism** adalah mekanisme yang memungkinkan model memberikan tingkat kepentingan berbeda pada bagian-bagian input ketika menghasilkan suatu representasi atau prediksi.

Misalnya, perhatikan kalimat:

> "Tolong bantu kami, rumah sudah terendam banjir."

Untuk mendeteksi permintaan bantuan darurat, kata atau frasa seperti `tolong`, `bantu`, dan `terendam banjir` mungkin relevan. Kata `kami` juga dapat membantu memahami konteks, tetapi kontribusinya bisa berbeda.

Attention membantu model menggabungkan informasi dari token-token input dengan bobot yang dipelajari. Namun, bobot attention **tidak otomatis menjadi penjelasan kausal** tentang keputusan model; interpretasinya perlu dilakukan dengan hati-hati.

### Analogi sederhana

Bayangkan Anda membaca sebuah kalimat untuk menentukan apakah penulis membutuhkan pertolongan. Anda mungkin lebih memperhatikan kata-kata yang menunjukkan kebutuhan, bahaya, atau dampak bencana. Attention melakukan proses pembobotan informasi yang serupa secara matematis.

### Gambaran alur

```text
Input tokens
     |
     v
Representasi vektor token
     |
     v
Hitung tingkat relevansi
     |
     v
Attention scores
     |
     v
Softmax -> attention weights
     |
     v
Gabungkan informasi berdasarkan bobot
     |
     v
Context-aware representation
```

---

## 2. Mengapa Attention Dibutuhkan?

Pada model sequence seperti RNN, informasi diproses secara berurutan. LSTM dan GRU membantu mempertahankan informasi lebih baik dibandingkan RNN sederhana, tetapi representasi akhir masih dapat menjadi ringkasan yang kurang memadai untuk input panjang.

Dalam arsitektur encoder-decoder klasik, decoder sering harus menggunakan representasi ringkas dari encoder. Attention memungkinkan decoder mengakses berbagai state encoder sesuai kebutuhan, bukan hanya mengandalkan satu ringkasan tetap.

Manfaat utama Attention:

- Memungkinkan model memilih informasi yang relevan dari beberapa posisi input.
- Menghubungkan informasi dari token yang berjauhan.
- Membantu model menangani hubungan kontekstual dalam sequence.
- Menjadi dasar bagi arsitektur Transformer.

**Catatan:** Attention bukan satu-satunya teknik yang dapat mengatasi keterbatasan sequence model. Perannya perlu dipahami sesuai arsitektur yang digunakan.

---

## 3. Konsep Query, Key, dan Value

Tiga komponen utama dalam bentuk attention yang umum digunakan adalah **Query (Q), Key (K), dan Value (V)**.

| Komponen | Intuisi | Peran |
| --- | --- | --- |
| Query (Q) | Informasi yang sedang dicari | Menentukan jenis informasi yang dibutuhkan |
| Key (K) | Label/ciri untuk dicocokkan | Digunakan untuk menghitung relevansi terhadap query |
| Value (V) | Informasi yang akan diambil | Digabungkan berdasarkan attention weights |

Analogi pencarian informasi:

- **Query:** pertanyaan yang ingin dijawab.
- **Key:** ciri yang digunakan untuk mencocokkan informasi.
- **Value:** isi informasi yang diambil.

Dalam Transformer, Q, K, dan V biasanya dibentuk dari representasi vektor token melalui proyeksi linear yang dipelajari selama training.

### Contoh konseptual

Kalimat:

```text
"Tolong bantu kami karena rumah terendam banjir."
```

Saat model membentuk representasi untuk token `bantu`, mekanisme attention dapat memberikan bobot pada token lain seperti `tolong`, `rumah`, `terendam`, dan `banjir`, bergantung pada representasi yang dipelajari.

Ini hanya ilustrasi. Bobot sebenarnya dihasilkan oleh model dan tidak dapat ditentukan hanya berdasarkan intuisi manusia.

---

## 4. Attention Score, Attention Weight, dan Context Vector

Tiga istilah ini tidak sama.

### 4.1 Attention score

Score menunjukkan tingkat kecocokan antara Query dan Key. Salah satu metode yang umum adalah dot product:

\[
s_i = QK_i^\top
\]

Score mentah dapat bernilai positif, nol, atau negatif. Nilainya belum berupa probabilitas.

### 4.2 Attention weight

Score biasanya dilewatkan ke fungsi **softmax** agar menjadi bobot yang jumlahnya satu:

\[
\alpha_i = \frac{\exp(s_i)}{\sum_j \exp(s_j)}
\]

Attention weight menunjukkan proporsi relatif yang digunakan untuk menggabungkan Value pada posisi berbeda.

### 4.3 Context vector

Context vector adalah gabungan berbobot dari Value:

\[
C = \sum_i \alpha_i V_i
\]

Artinya, Value dengan bobot lebih besar memberi kontribusi lebih besar terhadap context vector.

---

## 5. Contoh Perhitungan Manual

Anggap model sedang memproses sebuah token dan menghasilkan tiga attention score:

```text
Score:
Token A = 2.0
Token B = 1.0
Token C = 0.1
```

Softmax mengubah score menjadi bobot yang jumlahnya 1.

### Kode Python

```python
import numpy as np

scores = np.array([2.0, 1.0, 0.1])

# Stabilitas numerik: kurangi nilai maksimum sebelum exp
exp_scores = np.exp(scores - np.max(scores))
weights = exp_scores / exp_scores.sum()

print("Attention weights:", np.round(weights, 3))
print("Jumlah bobot:", round(weights.sum(), 3))
```

### Output

```text
Attention weights: [0.659 0.242 0.099]
Jumlah bobot: 1.0
```

Token A menerima bobot terbesar karena memiliki score tertinggi. Ini **tidak** berarti Token A pasti paling penting dalam semua konteks; hasil bergantung pada representasi dan score yang dihasilkan model.

Sekarang anggap Value setiap token adalah vektor dua dimensi:

```python
values = np.array([
    [1.0, 0.0],  # Token A
    [0.0, 1.0],  # Token B
    [1.0, 1.0],  # Token C
])

context = weights @ values

print("Context vector:", np.round(context, 3))
```

### Output

```text
Context vector: [0.758 0.341]
```

Context vector adalah kombinasi Value yang ditimbang oleh attention weights.

---

## 6. Scaled Dot-Product Attention

Transformer menggunakan bentuk yang dikenal sebagai **Scaled Dot-Product Attention**.

Rumusnya:

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

Keterangan:

- \(Q\): matriks Query.
- \(K\): matriks Key.
- \(V\): matriks Value.
- \(K^\top\): transpose dari matriks Key.
- \(d_k\): dimensi vektor Key.
- \(\sqrt{d_k}\): faktor skala agar dot product tidak terlalu besar ketika dimensinya tinggi.
- Softmax diterapkan pada score di setiap baris, sehingga bobot attention untuk setiap query berjumlah satu.

### Mengapa dibagi dengan \(\sqrt{d_k}\)?

Ketika dimensi vektor bertambah, dot product cenderung memiliki magnitudo lebih besar. Score yang terlalu besar dapat membuat softmax sangat tajam dan menyulitkan proses optimisasi. Faktor skala membantu menjaga rentang score.

### Bentuk tensor secara umum

Jika:

- \(Q\) memiliki bentuk `(batch, query_length, d_k)`,
- \(K\) memiliki bentuk `(batch, key_length, d_k)`,
- \(V\) memiliki bentuk `(batch, key_length, d_v)`,

maka:

- \(QK^\top\) memiliki bentuk `(batch, query_length, key_length)`,
- output attention memiliki bentuk `(batch, query_length, d_v)`.

---

## 7. Implementasi Scaled Dot-Product Attention dengan NumPy

Contoh ini menggunakan satu sequence tanpa batch agar perhitungannya mudah dilihat.

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Q: (query_length, d_k)
    K: (key_length, d_k)
    V: (key_length, d_v)
    """
    d_k = K.shape[-1]

    scores = Q @ K.T / np.sqrt(d_k)

    # Softmax pada sumbu key untuk setiap query
    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    output = weights @ V
    return output, weights, scores


Q = np.array([[1.0, 0.0]])
K = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])
V = np.array([
    [10.0, 0.0],
    [0.0, 10.0],
    [5.0, 5.0],
])

output, weights, scores = scaled_dot_product_attention(Q, K, V)

print("Scores:", np.round(scores, 3))
print("Weights:", np.round(weights, 3))
print("Output:", np.round(output, 3))
```

### Output yang diharapkan

```text
Scores: [[0.707 0.    0.707]]
Weights: [[0.401 0.198 0.401]]
Output: [[6.014 3.986]]
```

### Interpretasi

- Query paling cocok dengan Key pertama dan ketiga pada contoh ini.
- Keduanya mendapat bobot lebih tinggi daripada Key kedua.
- Output merupakan gabungan Value berdasarkan bobot tersebut.

Angka di atas berasal dari vektor buatan untuk demonstrasi, bukan hasil pelatihan model NLP.

---

## 8. Jenis Attention yang Perlu Diketahui

### 8.1 Encoder-Decoder Attention (Cross-Attention)

Dalam arsitektur encoder-decoder, Query dapat berasal dari decoder, sedangkan Key dan Value berasal dari output encoder.

```text
Encoder output -> Key, Value
                         \
                          -> Attention -> Decoder
                         /
Decoder state -> Query
```

Mekanisme ini membantu decoder memilih bagian input yang relevan ketika menghasilkan output. Contoh penggunaannya terdapat pada sejumlah model sequence-to-sequence.

### 8.2 Self-Attention

Dalam **Self-Attention**, Q, K, dan V dibentuk dari sequence yang sama, melalui proyeksi yang berbeda.

```text
Input token representations
          |
     ┌────┼────┐
     Q    K    V
     └────┼────┘
          |
    Self-Attention
          |
 Context-aware representations
```

Setiap token dapat menggabungkan informasi dari token lain di sequence tersebut, sesuai aturan mask yang digunakan.

Self-attention merupakan komponen penting dalam Transformer dan BERT.

### Perbedaan singkat

| Aspek | Cross-Attention | Self-Attention |
| --- | --- | --- |
| Query berasal dari | Sequence lain, misalnya decoder | Sequence yang sama |
| Key dan Value berasal dari | Umumnya output encoder | Sequence yang sama |
| Kegunaan umum | Menghubungkan output dengan input sumber | Membentuk representasi kontekstual dalam sequence |

---

## 9. Causal Attention dan Padding Mask

Dua konsep ini penting saat mempelajari Transformer.

### Causal mask

Causal attention mencegah posisi tertentu melihat token masa depan. Ini penting untuk model autoregresif yang memprediksi token berikutnya.

```text
Token 1 dapat melihat: Token 1
Token 2 dapat melihat: Token 1, Token 2
Token 3 dapat melihat: Token 1, Token 2, Token 3
```

BERT untuk pemahaman bahasa menggunakan mekanisme bidirectional dan umumnya tidak memakai causal mask seperti model autoregresif GPT.

### Padding mask

Batch teks sering memiliki panjang berbeda. Token padding ditambahkan agar bentuk tensor seragam. Padding mask mencegah model memberi perhatian pada posisi padding yang tidak seharusnya diperhitungkan.

**Jangan menyamakan kedua mask ini:** causal mask mengatur akses ke posisi masa depan, sedangkan padding mask mengabaikan posisi padding.

---

## 10. Multi-Head Attention

Satu mekanisme attention mempelajari satu jenis proyeksi Query, Key, dan Value pada satu head. **Multi-Head Attention** menjalankan beberapa head secara paralel, sehingga setiap head dapat mempelajari pola hubungan yang berbeda.

Alur sederhananya:

```text
Input
  |
  +--> Head 1: Attention(Q1, K1, V1) --+
  +--> Head 2: Attention(Q2, K2, V2) --+--> Concatenate
  +--> Head 3: Attention(Q3, K3, V3) --+        |
                                                v
                                          Linear layer
                                                |
                                                v
                                              Output
```

Secara matematis:

\[
\operatorname{head}_i =
\operatorname{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\]

\[
\operatorname{MultiHead}(Q,K,V)
=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O
\]

Keterangan:

- \(h\): jumlah attention heads.
- \(W_i^Q, W_i^K, W_i^V\): matriks proyeksi yang dipelajari.
- \(W^O\): proyeksi output setelah semua head digabungkan.

Jangan menganggap setiap head pasti memiliki fungsi semantik yang mudah diberi nama. Peran head perlu diuji, bukan ditebak hanya dari visualisasi.

---

## 11. Hubungan Attention dengan Transformer, BERT, dan IndoBERT

Perkembangan konsepnya dapat diringkas sebagai berikut:

```text
RNN / LSTM / GRU
        |
        v
Attention pada sequence model
        |
        v
Self-Attention + Multi-Head Attention
        |
        v
Transformer
        |
        v
BERT
        |
        v
IndoBERT
        |
        v
Fine-tuning untuk klasifikasi teks
```

Transformer menggunakan self-attention bersama komponen lain, seperti feed-forward network, residual connections, dan layer normalization.

BERT dibangun menggunakan tumpukan Transformer encoder. IndoBERT adalah model bahasa yang dilatih untuk bahasa Indonesia dengan pendekatan berbasis BERT.

Saat fine-tuning IndoBERT untuk klasifikasi tweet, Anda biasanya tidak perlu menghitung Q, K, dan V secara manual. Library seperti Hugging Face Transformers menangani operasi tersebut. Namun, memahami konsepnya membantu Anda menjelaskan arsitektur dan metodologi penelitian.

---

## 12. Relevansi untuk Penelitian Tweet Darurat Bencana

Contoh tweet:

```text
"Tolong bantu kami, rumah sudah terendam banjir."
```

Target klasifikasi:

```text
Emergency
Non-Emergency
```

Self-attention memungkinkan representasi setiap token dipengaruhi oleh konteks token lain. Misalnya, kata `banjir` dapat dipahami bersama `tolong bantu` dan `rumah sudah terendam`, bukan sebagai kata yang berdiri sendiri.

Namun, **Attention tidak otomatis menentukan label Emergency**. Prediksi kelas dihasilkan oleh model klasifikasi yang dilatih pada data berlabel. Kualitas prediksi juga dipengaruhi oleh kualitas anotasi, representasi data, distribusi kelas, dan evaluasi.

Untuk penelitian ini, setelah mempelajari Attention, lanjutkan ke:

1. Transformer encoder.
2. BERT dan tokenizer.
3. IndoBERT menggunakan Hugging Face.
4. Fine-tuning untuk klasifikasi Emergency/Non-Emergency.
5. Evaluasi dengan precision, recall, F1-score, dan confusion matrix.
6. Perhatian khusus pada **Recall kelas Emergency** serta penanganan class imbalance.

Bandingkan IndoBERT dengan baseline seperti TF-IDF + Logistic Regression dan TF-IDF + SVM agar eksperimen penelitian memiliki pembanding yang jelas.

---

## 13. Kesalahan Umum yang Perlu Dihindari

1. **Menganggap attention score dan attention weight sama.** Score adalah nilai kecocokan mentah; weight biasanya hasil softmax.
2. **Menganggap attention weight sebagai probabilitas label.** Weight menunjukkan pembobotan antar posisi input, bukan peluang kelas Emergency.
3. **Menganggap token dengan bobot terbesar selalu menjadi alasan utama prediksi.** Interpretasi attention tidak sesederhana itu.
4. **Menganggap semua Transformer memakai mask yang sama.** Model bidirectional dan autoregresif dapat menggunakan aturan mask berbeda.
5. **Mengira harus mengimplementasikan Attention dari nol untuk memakai IndoBERT.** Implementasi manual berguna untuk belajar; praktik fine-tuning dapat memakai library.
6. **Langsung melompat ke rumus tanpa memahami bentuk tensor.** Biasakan mencatat dimensi Q, K, V, score, weight, dan output.

---

## 14. Latihan

### Latihan 1 — Konsep

Jelaskan dengan kata-kata sendiri:

1. Apa fungsi Query?
2. Apa perbedaan Key dan Value?
3. Mengapa attention score dilewatkan ke softmax?
4. Apa fungsi context vector?
5. Apa perbedaan self-attention dan cross-attention?

### Latihan 2 — Perhitungan

Diberikan score:

```python
scores = np.array([0.0, 1.0, 2.0])
```

1. Hitung softmax.
2. Pastikan jumlah bobot mendekati `1.0`.
3. Jelaskan token mana yang menerima bobot tertinggi.

### Latihan 3 — Modifikasi kode

Pada implementasi NumPy di atas:

1. Ubah nilai Query.
2. Jalankan kembali kode.
3. Bandingkan attention weights dan context vector.
4. Jelaskan mengapa output berubah.

### Latihan 4 — Hubungkan ke penelitian

Buat tiga contoh tweet yang berbeda:

- Satu jelas merupakan permintaan bantuan darurat.
- Satu berisi informasi bencana tanpa permintaan bantuan eksplisit.
- Satu tidak berkaitan dengan keadaan darurat.

Jelaskan mengapa konteks keseluruhan kalimat diperlukan untuk mengklasifikasikannya. Jangan menganggap kata `banjir` saja selalu cukup untuk menentukan label.

---

## 15. Mini Project — Visualisasi Attention Sederhana

Gunakan kode scaled dot-product attention pada bagian 7.

Kembangkan program agar menampilkan:

- attention weights untuk setiap Query,
- context vector,
- perubahan bobot saat Query diubah.

Pada tahap ini gunakan vektor buatan. Visualisasi ini bertujuan memahami perhitungan, bukan menjelaskan perilaku IndoBERT yang telah dilatih.

**Kriteria selesai:**

- [ ] Dapat menjelaskan Q, K, dan V.
- [ ] Dapat membedakan score, weight, dan context vector.
- [ ] Dapat menjalankan kode NumPy dan menjelaskan outputnya.
- [ ] Memahami scaled dot-product attention.
- [ ] Memahami self-attention dan multi-head attention.
- [ ] Memahami fungsi causal mask dan padding mask.
- [ ] Dapat menjelaskan hubungan Attention → Transformer → BERT → IndoBERT.

---

## 16. Ringkasan

| Konsep | Inti pemahaman |
| --- | --- |
| Query | Representasi informasi yang dicari |
| Key | Representasi yang dicocokkan dengan Query |
| Value | Informasi yang digabungkan |
| Attention score | Skor kecocokan mentah |
| Attention weight | Bobot hasil normalisasi, biasanya softmax |
| Context vector | Gabungan Value berdasarkan bobot |
| Scaled dot-product | Attention dengan score \(QK^\top\) yang dibagi \(\sqrt{d_k}\) |
| Self-attention | Q, K, V berasal dari sequence yang sama |
| Cross-attention | Query dan Key/Value berasal dari sequence berbeda |
| Multi-head attention | Beberapa attention head berjalan paralel |
| Mask | Membatasi posisi yang dapat diperhatikan |

### Langkah selanjutnya

Setelah materi ini, pelajari **Transformer Architecture**. Fokus pada embedding, positional information, multi-head self-attention, feed-forward network, residual connection, dan layer normalization. Setelah itu, lanjutkan ke BERT dan IndoBERT.
