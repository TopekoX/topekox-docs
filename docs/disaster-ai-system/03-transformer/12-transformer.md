# Transformer untuk Natural Language Processing (NLP)

## Tujuan Pembelajaran

Setelah mempelajari materi ini, Anda diharapkan mampu:

1. Menjelaskan alasan Transformer dikembangkan.
2. Memahami arsitektur Transformer dan komponen utamanya.
3. Menjelaskan self-attention, multi-head attention, positional encoding, encoder, dan decoder.
4. Memahami perbedaan Transformer encoder, decoder, dan encoder-decoder.
5. Menjelaskan hubungan Transformer dengan BERT dan IndoBERT.
6. Mengimplementasikan ilustrasi sederhana self-attention menggunakan Python.

---

## 1. Apa Itu Transformer?

**Transformer** adalah arsitektur deep learning yang dirancang untuk memproses data berurutan, termasuk teks. Arsitektur ini diperkenalkan dalam paper *Attention Is All You Need* (Vaswani et al., 2017).

Sebelum Transformer, banyak model NLP menggunakan Recurrent Neural Network (RNN), LSTM, atau GRU. Model tersebut memproses token secara berurutan. Transformer menggunakan mekanisme **attention** untuk menghubungkan token-token dalam suatu urutan dan dapat memproses token secara paralel selama tahap pelatihan.

Contoh kalimat:

> Saya membutuhkan bantuan karena rumah saya terkena banjir.

Untuk memahami konteks kata *bantuan*, model mungkin perlu mempertimbangkan kata *membutuhkan*, *rumah*, dan *banjir*. Self-attention memungkinkan setiap token memberi bobot pada token lain yang relevan.

### 1.1 Mengapa Transformer dibutuhkan?

Beberapa keterbatasan model berurutan:

- Pemrosesan RNN/LSTM sulit diparalelkan sepenuhnya karena suatu langkah bergantung pada langkah sebelumnya.
- Informasi dari token yang berjauhan dapat sulit dipertahankan pada urutan panjang.
- Pelatihan pada data besar dapat menjadi mahal dari sisi waktu.

Transformer membantu mengatasi keterbatasan tersebut melalui attention dan pemrosesan paralel. Namun, Transformer juga dapat membutuhkan memori besar, terutama ketika urutan input panjang.

### 1.2 Gambaran umum

```text
Teks
  ↓
Tokenisasi
  ↓
Token Embedding
  ↓
Informasi Posisi
  ↓
Transformer Layer
  ├── Self-Attention
  ├── Feed-Forward Network
  ├── Residual Connection
  └── Layer Normalization
  ↓
Representasi Kontekstual
  ↓
Head tugas (misalnya klasifikasi)
  ↓
Prediksi
```

---

## 2. Konsep Dasar yang Perlu Dipahami

Sebelum mempelajari detail Transformer, pahami istilah berikut.

| Istilah | Penjelasan |
|---|---|
| Token | Unit teks yang diproses model; dapat berupa kata, bagian kata (subword), atau simbol. |
| Tokenizer | Mengubah teks menjadi token dan biasanya memetakan token menjadi ID. |
| Embedding | Vektor numerik yang mewakili token. |
| Hidden state | Representasi vektor token setelah diproses suatu layer. |
| Attention | Mekanisme untuk menghitung seberapa besar suatu token perlu memperhatikan token lain. |
| Sequence length | Jumlah token dalam input setelah tokenisasi, termasuk token khusus jika ada. |
| Batch | Sekumpulan contoh yang diproses bersama. |

Komputer tidak memproses kata sebagai makna manusia secara langsung. Teks diubah menjadi ID token, lalu ID tersebut digunakan untuk mengambil embedding numerik.

---

## 3. Token Embedding dan Informasi Posisi

### 3.1 Token embedding

Misalnya teks:

```text
"tolong bantu kami"
```

Tokenizer dapat menghasilkan token dan ID seperti berikut. Hasil sebenarnya bergantung pada tokenizer yang digunakan.

| Token ilustratif | ID ilustratif |
|---|---:|
| tolong | 105 |
| bantu | 872 |
| kami | 246 |

ID token hanyalah indeks, bukan ukuran makna. Model menggunakan tabel embedding untuk mengubah setiap ID menjadi vektor.

Secara sederhana:

\[
E \in \mathbb{R}^{n \times d_{\text{model}}}
\]

Keterangan:

- \(n\): jumlah token dalam urutan.
- \(d_{\text{model}}\): dimensi embedding/model.
- \(E\): matriks embedding seluruh token.

### 3.2 Mengapa informasi posisi diperlukan?

Self-attention dasar tidak otomatis mengetahui urutan token. Kalimat berikut memiliki kata yang sama tetapi urutan berbeda:

- `anjing mengejar kucing`
- `kucing mengejar anjing`

Karena urutan mengubah makna, Transformer memerlukan informasi posisi.

Pada Transformer asli, **positional encoding** ditambahkan ke embedding token. Salah satu bentuk sinusoidalnya adalah:

\[
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]

\[
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]

Keterangan:

- \(pos\): posisi token dalam urutan.
- \(i\): indeks dimensi.
- \(d_{\text{model}}\): dimensi representasi model.

Secara ringkas:

\[
X = E + PE
\]

Transformer modern tidak semuanya memakai positional encoding sinusoidal. Sebagian menggunakan embedding posisi yang dipelajari atau metode posisi lain, seperti rotary positional embedding (RoPE).

---

## 4. Attention: Intuisi Dasar

Attention membantu model membangun representasi token dengan mempertimbangkan token lain.

Contoh:

> Korban meminta bantuan karena rumahnya terendam banjir.

Ketika membentuk representasi kata *bantuan*, model dapat memberikan bobot berbeda kepada kata *korban*, *meminta*, *rumahnya*, dan *banjir*. Bobot tersebut dipelajari dari data; contoh ini hanya ilustrasi, bukan hasil interpretasi model yang sebenarnya.

### 4.1 Query, Key, dan Value

Setiap representasi token diproyeksikan menjadi tiga vektor:

- **Query (Q):** informasi tentang apa yang dicari token.
- **Key (K):** informasi yang digunakan untuk mencocokkan token dengan query.
- **Value (V):** informasi yang digabungkan berdasarkan bobot attention.

Query dibandingkan dengan key untuk memperoleh skor relevansi. Skor tersebut dinormalisasi menjadi bobot, lalu bobot digunakan untuk menggabungkan value.

### 4.2 Scaled Dot-Product Attention

Rumus inti attention:

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
\]

Keterangan:

- \(Q\): matriks query.
- \(K\): matriks key.
- \(V\): matriks value.
- \(d_k\): dimensi key.
- \(QK^\top\): skor kemiripan antara query dan key.
- softmax: mengubah skor menjadi bobot yang jumlahnya 1 pada setiap baris.

Pembagian dengan \(\sqrt{d_k}\) menjaga skala skor agar tidak terlalu besar ketika dimensi key meningkat.

### 4.3 Alur perhitungan

```text
Query dan Key
      ↓
Skor QKᵀ
      ↓
Skala dengan √dₖ
      ↓
Softmax
      ↓
Bobot Attention
      ↓
Kalikan dengan Value
      ↓
Representasi Kontekstual
```

Attention bukan sekadar mencari kata yang sama. Bobotnya dihasilkan dari proyeksi yang dipelajari dan konteks input.

---

## 5. Self-Attention

Pada **self-attention**, Q, K, dan V berasal dari representasi urutan input yang sama. Setiap token dapat memperhatikan token lain dalam urutan tersebut.

Untuk input \(X\):

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
\]

Kemudian:

\[
\operatorname{SelfAttention}(X)
=
\operatorname{softmax}\left(
\frac{XW_Q(XW_K)^\top}{\sqrt{d_k}}
\right)XW_V
\]

\(W_Q\), \(W_K\), dan \(W_V\) adalah matriks bobot yang dipelajari selama pelatihan.

### Contoh konseptual

Input:

> Saya membutuhkan bantuan karena banjir.

Self-attention memungkinkan representasi setiap token menggabungkan informasi dari token lain. Pada encoder BERT, perhatian umumnya dapat diarahkan ke token di sebelah kiri maupun kanan, sehingga konteksnya bersifat bidirectional.

**Catatan:** Bobot attention tidak otomatis merupakan penjelasan lengkap tentang alasan model membuat prediksi.

---

## 6. Masking

Masking mengatur posisi mana yang boleh diperhatikan.

Dua jenis yang umum:

1. **Padding mask:** mencegah model memperhatikan token padding yang hanya digunakan untuk menyamakan panjang urutan dalam satu batch.
2. **Causal mask:** mencegah token melihat token masa depan. Masking ini umum pada model autoregresif seperti GPT ketika memprediksi token berikutnya.

BERT menggunakan masking untuk menangani padding, tetapi tidak menggunakan causal mask standar saat menjalankan encoder untuk memahami teks.

---

## 7. Multi-Head Attention

Satu attention head dapat mempelajari satu jenis hubungan. **Multi-head attention** menjalankan beberapa attention head secara paralel sehingga model dapat membentuk representasi dari beberapa pola hubungan.

Rumus ringkas:

\[
\operatorname{head}_i =
\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V)
\]

\[
\operatorname{MultiHead}(Q,K,V)
=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O
\]

Keterangan:

- \(h\): jumlah attention head.
- \(W_i^Q, W_i^K, W_i^V\): proyeksi untuk head ke-\(i\).
- \(W^O\): proyeksi keluaran setelah semua head digabungkan.

Ilustrasi:

```text
Input
  ├── Attention Head 1 ──┐
  ├── Attention Head 2 ──┤
  ├── Attention Head 3 ──┤
  └── Attention Head h ──┘
             ↓
          Concatenate
             ↓
        Linear Projection
             ↓
           Output
```

Jangan menganggap setiap head pasti memiliki fungsi linguistik tertentu. Peran head dapat tumpang tindih dan tidak selalu mudah ditafsirkan.

---

## 8. Feed-Forward Network

Setelah attention, setiap posisi token diproses oleh **position-wise feed-forward network**. Pada Transformer standar, jaringan ini umumnya terdiri dari dua transformasi linear dengan fungsi aktivasi di antaranya.

\[
\operatorname{FFN}(x)=\sigma(xW_1+b_1)W_2+b_2
\]

Keterangan:

- \(W_1, W_2\): matriks bobot.
- \(b_1, b_2\): bias.
- \(\sigma\): fungsi aktivasi, misalnya ReLU pada Transformer asli.

Attention menggabungkan informasi antar-token, sedangkan FFN mengolah representasi setiap posisi secara terpisah. Model modern dapat menggunakan variasi seperti GELU atau gated feed-forward network.

---

## 9. Residual Connection dan Layer Normalization

Transformer menggunakan residual connection dan layer normalization untuk membantu optimisasi dan stabilitas pelatihan.

Secara konseptual:

\[
Y=\operatorname{LayerNorm}(X+\operatorname{Sublayer}(X))
\]

Persamaan tersebut menggambarkan salah satu susunan yang dikenal sebagai post-layer normalization. Implementasi lain, termasuk banyak model modern, menerapkan layer normalization sebelum sublayer (*pre-LN*).

- **Residual connection:** menambahkan input sublayer ke keluarannya.
- **Layer normalization:** menormalkan aktivasi pada dimensi fitur untuk membantu kestabilan pelatihan.

---

## 10. Encoder dan Decoder

Transformer asli terdiri dari tumpukan encoder dan decoder.

### 10.1 Encoder

Encoder membangun representasi kontekstual dari input. Setiap layer umumnya memiliki:

1. Multi-head self-attention.
2. Residual connection dan layer normalization.
3. Feed-forward network.
4. Residual connection dan layer normalization.

### 10.2 Decoder

Decoder Transformer asli memiliki:

1. Masked self-attention.
2. Encoder-decoder attention yang memperhatikan keluaran encoder.
3. Feed-forward network.
4. Residual connection dan layer normalization.

Decoder digunakan untuk menghasilkan keluaran secara bertahap pada tugas seperti penerjemahan.

### 10.3 Perbedaan keluarga model

| Keluarga | Pola arsitektur | Contoh penggunaan |
|---|---|---|
| Encoder-only | Tumpukan encoder | Klasifikasi teks, NER, pemahaman teks |
| Decoder-only | Tumpukan decoder dengan causal masking | Generasi teks |
| Encoder-decoder | Encoder dan decoder | Terjemahan, summarization, text-to-text |

Contoh keluarga model:

- **BERT/IndoBERT:** encoder-only.
- **GPT:** decoder-only.
- **T5:** encoder-decoder.

Untuk klasifikasi tweet berbahasa Indonesia, model encoder-only seperti IndoBERT merupakan pilihan yang relevan untuk dievaluasi.

---

## 11. Bagaimana Transformer Memproses Teks?

Secara umum, pipeline Transformer untuk pemahaman teks:

```text
"Tolong bantu kami karena banjir"
                ↓
            Tokenizer
                ↓
      Token IDs + Attention Mask
                ↓
       Token Embedding + Posisi
                ↓
         Transformer Encoder
                ↓
     Representasi Kontekstual
                ↓
      Classification Head (jika ada)
                ↓
       Probabilitas / Label
```

Detail seperti token khusus, metode posisi, dan bentuk keluaran bergantung pada model serta tokenizer yang digunakan.

---

## 12. Contoh Python: Ilustrasi Self-Attention

Contoh berikut memperlihatkan ide inti self-attention menggunakan NumPy. Contoh ini **bukan implementasi Transformer lengkap**: tidak memiliki parameter yang dipelajari, multi-head attention, positional encoding, FFN, residual connection, atau training.

```python
import numpy as np

# Tiga token dengan representasi vektor ilustratif.
X = np.array([
    [1.0, 0.0],  # token 1
    [0.0, 1.0],  # token 2
    [1.0, 1.0],  # token 3
])

# Agar mudah dipahami, gunakan matriks proyeksi identitas.
W_Q = np.eye(2)
W_K = np.eye(2)
W_V = np.eye(2)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

# Hitung scaled dot-product attention.
scores = Q @ K.T / np.sqrt(K.shape[1])

# Softmax per baris agar menjadi bobot attention.
scores_shifted = scores - scores.max(axis=1, keepdims=True)
weights = np.exp(scores_shifted)
weights = weights / weights.sum(axis=1, keepdims=True)

# Gabungkan value berdasarkan bobot attention.
output = weights @ V

print("Attention weights:")
print(np.round(weights, 3))

print("\nContextual representations:")
print(np.round(output, 3))
```

### Contoh output

```text
Attention weights:
[[0.401 0.198 0.401]
 [0.198 0.401 0.401]
 [0.248 0.248 0.503]]

Contextual representations:
[[0.802 0.599]
 [0.599 0.802]
 [0.752 0.752]]
```

### Interpretasi

- Setiap baris pada `weights` mewakili bobot perhatian dari satu token terhadap semua token.
- Jumlah bobot pada setiap baris mendekati 1.
- `output` adalah kombinasi berbobot dari representasi `V`.
- Karena contoh menggunakan matriks identitas dan data buatan, hasil ini hanya ilustrasi matematis, bukan perilaku model NLP terlatih.

---

## 13. Hubungan Transformer dengan BERT dan IndoBERT

BERT dibangun menggunakan tumpukan Transformer encoder. IndoBERT adalah model bahasa yang dilatih untuk bahasa Indonesia dan menggunakan keluarga arsitektur BERT.

Hubungan sederhananya:

```text
Transformer
    ↓
Transformer Encoder
    ↓
BERT
    ↓
IndoBERT
    ↓
Fine-tuning untuk klasifikasi
    ↓
Prediksi label teks
```

Dalam fine-tuning klasifikasi:

1. Teks ditokenisasi menggunakan tokenizer yang cocok dengan checkpoint model.
2. Token IDs dan attention mask dimasukkan ke IndoBERT.
3. Model menghasilkan representasi kontekstual.
4. Classification head menghasilkan skor kelas.
5. Model dilatih pada dataset berlabel untuk menyesuaikan prediksi dengan tugas.

Untuk penelitian tweet bantuan darurat, model dapat dilatih membedakan kelas seperti `Emergency` dan `Non-Emergency`. Kualitas hasil tetap bergantung pada mutu dataset, konsistensi anotasi, pembagian data, dan evaluasi.

---

## 14. Hal yang Perlu Diingat

1. Transformer adalah arsitektur, bukan satu model tunggal.
2. Self-attention menghubungkan informasi antar-token.
3. Positional information membantu model mempertimbangkan urutan.
4. Multi-head attention memungkinkan beberapa proyeksi attention dipelajari secara paralel.
5. Encoder-only, decoder-only, dan encoder-decoder memiliki pola penggunaan yang berbeda.
6. BERT dan IndoBERT termasuk model encoder-only.
7. Untuk klasifikasi, Transformer biasanya dipasangkan dengan classification head dan di-fine-tune.
8. Attention weights tidak boleh langsung dianggap sebagai penjelasan kausal atas prediksi.
9. Transformer dapat memerlukan memori dan komputasi besar untuk input panjang.

---

## 15. Latihan

### Latihan 1 — Konsep

Jawab dengan kalimat sendiri:

1. Mengapa Transformer dikembangkan?
2. Apa perbedaan token dan embedding?
3. Apa fungsi Query, Key, dan Value?
4. Mengapa Transformer memerlukan informasi posisi?
5. Apa perbedaan self-attention dan multi-head attention?
6. Apa fungsi feed-forward network?
7. Apa perbedaan encoder-only dan decoder-only?
8. Mengapa IndoBERT sesuai untuk dicoba pada klasifikasi teks bahasa Indonesia?

### Latihan 2 — Praktik

1. Jalankan kode NumPy pada bagian 12.
2. Pastikan jumlah bobot pada setiap baris `weights` mendekati 1.
3. Ubah salah satu vektor pada `X`, kemudian jalankan kembali.
4. Amati perubahan bobot attention dan keluaran.
5. Jelaskan mengapa perubahan input mengubah representasi keluaran.

### Latihan 3 — Persiapan Penelitian

Gunakan kalimat berikut:

> Tolong bantu kami, rumah terendam banjir.

Jelaskan alur pemrosesannya dari teks mentah, tokenisasi, representasi Transformer, hingga klasifikasi `Emergency` atau `Non-Emergency`. Tidak perlu menebak bobot attention aktual; fokus pada alur sistem.

---

## 16. Checklist Kelulusan

- [ ] Saya memahami alasan Transformer digunakan dalam NLP.
- [ ] Saya dapat menjelaskan embedding dan informasi posisi.
- [ ] Saya memahami rumus scaled dot-product attention.
- [ ] Saya dapat menjelaskan Query, Key, dan Value.
- [ ] Saya memahami multi-head attention.
- [ ] Saya memahami FFN, residual connection, dan layer normalization.
- [ ] Saya dapat membedakan encoder-only, decoder-only, dan encoder-decoder.
- [ ] Saya memahami hubungan Transformer, BERT, dan IndoBERT.
- [ ] Saya dapat menjalankan dan menjelaskan ilustrasi self-attention NumPy.

Jika sebagian besar checklist sudah terpenuhi, lanjutkan ke materi **BERT**, lalu praktik tokenizer dan fine-tuning IndoBERT menggunakan Hugging Face.

---

## Referensi

1. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT.
3. Hugging Face, *Transformers Documentation*: https://huggingface.co/docs/transformers
