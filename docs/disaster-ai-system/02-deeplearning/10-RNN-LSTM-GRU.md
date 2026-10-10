# Materi RNN → LSTM → GRU untuk NLP

Materi ini mengikuti roadmap belajar NLP dari dasar menuju Transformer dan IndoBERT. Fokusnya adalah memahami bagaimana model memproses urutan teks, mengapa RNN memiliki keterbatasan, bagaimana LSTM dan GRU mengatasinya, serta bagaimana menjalankan contoh kode sederhana.

Konteks penelitian yang digunakan sebagai ilustrasi adalah klasifikasi teks media sosial terkait permintaan bantuan darurat saat bencana. RNN, LSTM, dan GRU dipelajari sebagai fondasi konseptual sebelum masuk ke Attention, Transformer, dan IndoBERT.

## Tujuan Pembelajaran

Setelah mempelajari materi ini, Anda diharapkan mampu:

1. Menjelaskan konsep sequence dalam NLP.
2. Memahami cara kerja Recurrent Neural Network (RNN).
3. Menjelaskan hidden state dan pemrosesan token secara berurutan.
4. Memahami masalah vanishing gradient dan keterbatasan RNN.
5. Menjelaskan bagaimana LSTM menggunakan cell state dan gates.
6. Menjelaskan cara kerja GRU dan perbedaannya dari LSTM.
7. Menjalankan contoh kode PyTorch sederhana dan membaca bentuk output-nya.
8. Memilih eksperimen sederhana untuk membandingkan RNN, LSTM, dan GRU.

## 1. Mengapa Model Sequence Dibutuhkan?

Teks bukan sekadar kumpulan kata. Urutan kata dapat mengubah makna kalimat.

Contoh:

- “Korban membutuhkan bantuan.”
- “Korban tidak membutuhkan bantuan.”

Kata *tidak* mengubah makna kalimat. Model NLP idealnya dapat memanfaatkan hubungan antarkata, termasuk informasi yang muncul lebih awal dalam kalimat.

Pada pendekatan seperti Bag of Words atau TF-IDF, teks direpresentasikan sebagai fitur numerik. Representasi tersebut berguna untuk baseline machine learning, tetapi tidak secara langsung memodelkan urutan token seperti model sequence.

### Istilah penting

| Istilah | Penjelasan |
|---|---|
| Sequence | Urutan elemen yang diproses, misalnya token dalam kalimat. |
| Token | Unit teks, misalnya kata atau subword. |
| Time step | Posisi langkah pemrosesan dalam sequence. |
| Embedding | Vektor numerik yang mewakili token. |
| Hidden state | Representasi internal yang dibawa dari satu time step ke time step berikutnya. |
| Cell state | Memori internal khusus pada LSTM. |
| Gate | Mekanisme yang mengatur informasi yang disimpan, dibuang, atau diteruskan. |

## 2. Gambaran Umum: RNN → LSTM → GRU

Perkembangan sederhananya:

```text
Token sequence
      ↓
     RNN
      ↓
Masalah membawa informasi jauh dapat muncul
      ↓
     LSTM
      ↓
Memori dan pengaturan informasi melalui gates
      ↓
      GRU
      ↓
Alternatif dengan struktur gate yang lebih ringkas
```

RNN, LSTM, dan GRU sama-sama memproses data berurutan. LSTM dan GRU merupakan jenis recurrent network yang dirancang untuk membantu menangani ketergantungan sequence yang lebih panjang dibandingkan RNN sederhana. Namun, ketiganya tetap memproses sequence secara berurutan, sehingga pelatihan paralelnya lebih terbatas daripada Transformer.

## 3. Recurrent Neural Network (RNN)

### 3.1 Apa itu RNN?

Recurrent Neural Network adalah jenis neural network yang membawa informasi dari langkah sebelumnya melalui *hidden state*. Pada setiap time step, model menerima input saat ini dan hidden state sebelumnya untuk menghitung hidden state baru.

Contoh kalimat:

```text
"Tolong bantu korban banjir"
```

Secara konseptual, pemrosesannya berlangsung seperti berikut:

```text
Tolong → bantu → korban → banjir
   h1      h2       h3       h4
```

Setiap hidden state dapat membawa sebagian informasi dari token-token yang telah diproses.

### 3.2 Rumus dasar RNN

Salah satu bentuk sederhana persamaan RNN adalah:

\[
h_t = \tanh(W_x x_t + W_h h_{t-1} + b_h)
\]

Keterangan:

- \(x_t\): input atau embedding pada time step ke-\(t\).
- \(h_{t-1}\): hidden state dari langkah sebelumnya.
- \(W_x\): bobot untuk input.
- \(W_h\): bobot recurrent untuk hidden state sebelumnya.
- \(b_h\): bias.
- \(\tanh\): fungsi aktivasi.
- \(h_t\): hidden state yang baru dihitung.

Untuk klasifikasi sequence, representasi akhir atau gabungan representasi hidden state dapat diteruskan ke layer klasifikasi.

### 3.3 Kelebihan RNN

- Arsitektur relatif sederhana.
- Memperhitungkan urutan data.
- Cocok sebagai model pembelajaran dan baseline sequence sederhana.

### 3.4 Keterbatasan RNN

- Pemrosesan token berlangsung berurutan.
- Informasi dari token yang jauh sebelumnya dapat sulit dipertahankan.
- Dapat mengalami *vanishing gradient* atau *exploding gradient* saat training.
- Kinerja dapat menurun pada ketergantungan jangka panjang.

### 3.5 Apa itu vanishing gradient?

Saat model dilatih menggunakan backpropagation through time (BPTT), gradien mengalir melewati banyak time step. Dalam kondisi tertentu, gradien dapat menjadi sangat kecil ketika melewati banyak operasi berulang. Akibatnya, parameter yang berhubungan dengan bagian awal sequence menerima sinyal pembelajaran yang lemah.

Dampaknya: model mungkin kesulitan mempelajari hubungan antara token yang berjauhan.

*Exploding gradient* adalah kondisi kebalikannya, ketika gradien menjadi sangat besar dan training tidak stabil. Teknik seperti gradient clipping dapat membantu mengatasi exploding gradient, tetapi tidak menghilangkan seluruh keterbatasan RNN.

## 4. Contoh Kode RNN dengan PyTorch

Contoh berikut menggunakan input numerik buatan agar fokusnya pada bentuk tensor dan alur model, bukan pada proses tokenisasi teks. Model ini belum dilatih untuk menghasilkan prediksi bermakna.

### 4.1 Instalasi

Jika PyTorch belum tersedia, instal sesuai petunjuk resmi untuk sistem Anda. Di lingkungan notebook, biasanya dapat dimulai dengan:

```bash
pip install torch
```

### 4.2 Membuat RNN sederhana

```python
import torch
import torch.nn as nn

# Contoh tensor:
# batch_size = 2, sequence_length = 4, input_size = 3
x = torch.tensor([
    [[0.1, 0.2, 0.3],
     [0.2, 0.1, 0.4],
     [0.3, 0.5, 0.2],
     [0.4, 0.2, 0.1]],

    [[0.2, 0.3, 0.1],
     [0.1, 0.4, 0.2],
     [0.5, 0.2, 0.3],
     [0.3, 0.1, 0.4]]
], dtype=torch.float32)

rnn = nn.RNN(
    input_size=3,
    hidden_size=5,
    batch_first=True
)

output, h_n = rnn(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Final hidden state shape:", h_n.shape)
```

### 4.3 Contoh output

```text
Input shape: torch.Size([2, 4, 3])
Output shape: torch.Size([2, 4, 5])
Final hidden state shape: torch.Size([1, 2, 5])
```

Nilai tensor aktual di `output` dan `h_n` bergantung pada parameter model yang diinisialisasi PyTorch. Bentuk tensor di atas menunjukkan struktur dimensinya.

### 4.4 Memahami shape

| Tensor | Shape | Makna |
|---|---|---|
| `x` | `(2, 4, 3)` | 2 contoh, 4 time step, 3 fitur input per time step. |
| `output` | `(2, 4, 5)` | Hidden output untuk setiap time step, dengan 5 fitur hidden. |
| `h_n` | `(1, 2, 5)` | Hidden state terakhir untuk 1 layer/direction, 2 contoh, dan 5 fitur hidden. |

Dengan `batch_first=True`, bentuk input RNN adalah `(batch, sequence, features)`.

## 5. Long Short-Term Memory (LSTM)

### 5.1 Mengapa LSTM dikembangkan?

LSTM merupakan jenis RNN yang dirancang untuk membantu mempertahankan informasi yang relevan dalam rentang sequence yang lebih panjang. LSTM menambahkan *cell state* dan mekanisme *gating* untuk mengatur aliran informasi.

Gambaran sederhana:

```text
Input token
    ↓
Forget gate ── menentukan informasi lama yang dipertahankan
    ↓
Input gate ─── menentukan informasi baru yang disimpan
    ↓
Cell state ─── jalur memori
    ↓
Output gate ── menentukan informasi yang dikeluarkan
    ↓
Hidden state
```

### 5.2 Tiga gate pada LSTM

**Forget gate**

Mengatur bagian cell state sebelumnya yang akan dipertahankan atau dilupakan.

**Input gate**

Mengatur informasi baru yang akan ditambahkan ke cell state.

**Output gate**

Mengatur bagian informasi dari cell state yang digunakan untuk membentuk hidden state saat ini.

### 5.3 Persamaan konseptual LSTM

Persamaan berikut merupakan bentuk umum LSTM:

\[
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
\]

\[
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
\]

\[
\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)
\]

\[
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
\]

\[
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
\]

\[
h_t = o_t \odot \tanh(c_t)
\]

Keterangan:

- \(f_t\): forget gate.
- \(i_t\): input gate.
- \(\tilde{c}_t\): kandidat informasi baru.
- \(c_t\): cell state.
- \(o_t\): output gate.
- \(h_t\): hidden state.
- \(\sigma\): fungsi sigmoid, yang menghasilkan nilai antara 0 dan 1.
- \(\odot\): perkalian elemen demi elemen.

Nilai gate tidak berarti keputusan bahasa yang eksplisit seperti manusia; gate adalah mekanisme numerik yang dipelajari selama training.

### 5.4 Kelebihan dan keterbatasan LSTM

**Kelebihan:**

- Memiliki jalur cell state untuk membawa informasi.
- Gate membantu mengatur informasi yang dipertahankan dan diperbarui.
- Sering lebih efektif daripada RNN sederhana pada sequence yang memiliki ketergantungan lebih panjang.

**Keterbatasan:**

- Struktur lebih kompleks daripada RNN sederhana.
- Memiliki lebih banyak parameter daripada GRU dengan ukuran input dan hidden yang sebanding pada implementasi umum.
- Tetap memproses time step secara berurutan.
- Tidak selalu mengungguli GRU atau model Transformer pada setiap dataset.

## 6. Contoh Kode LSTM dengan PyTorch

```python
import torch
import torch.nn as nn

x = torch.randn(2, 4, 3)

lstm = nn.LSTM(
    input_size=3,
    hidden_size=5,
    batch_first=True
)

output, (h_n, c_n) = lstm(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Final hidden state shape:", h_n.shape)
print("Final cell state shape:", c_n.shape)
```

### Contoh output

```text
Input shape: torch.Size([2, 4, 3])
Output shape: torch.Size([2, 4, 5])
Final hidden state shape: torch.Size([1, 2, 5])
Final cell state shape: torch.Size([1, 2, 5])
```

Nilai tensor aktual bersifat dinamis, sedangkan shape di atas mengikuti konfigurasi model.

### Perbedaan output RNN dan LSTM

- `output`: hidden state untuk setiap time step.
- `h_n`: hidden state terakhir dari setiap layer/direction.
- `c_n`: cell state terakhir dari setiap layer/direction; tersedia pada LSTM, bukan RNN sederhana.

## 7. Gated Recurrent Unit (GRU)

### 7.1 Apa itu GRU?

GRU adalah arsitektur recurrent yang juga menggunakan gate untuk mengatur informasi. GRU lebih ringkas daripada LSTM dalam implementasi standar karena tidak memiliki cell state terpisah dan menggunakan dua gate utama.

Dua gate pada GRU:

- **Update gate**: mengatur seberapa banyak informasi sebelumnya dipertahankan dibandingkan informasi kandidat yang baru.
- **Reset gate**: mengatur bagaimana informasi hidden state sebelumnya digunakan saat menghitung kandidat hidden state baru.

Secara konseptual:

```text
Input token + Hidden state sebelumnya
                  ↓
             Reset gate
                  ↓
           Kandidat hidden state
                  ↓
             Update gate
                  ↓
             Hidden state baru
```

### 7.2 Persamaan konseptual GRU

Salah satu bentuk persamaan GRU yang umum adalah:

\[
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
\]

\[
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
\]

\[
\tilde{h}_t = \tanh(W_h x_t + U_h(r_t \odot h_{t-1}) + b_h)
\]

\[
h_t = (1-z_t)\odot \tilde{h}_t + z_t \odot h_{t-1}
\]

Keterangan:

- \(r_t\): reset gate.
- \(z_t\): update gate.
- \(\tilde{h}_t\): kandidat hidden state.
- \(h_t\): hidden state baru.

Catatan: beberapa referensi menggunakan konvensi persamaan atau penamaan gate yang berbeda. Inti konsepnya tetap bahwa GRU mengatur perpaduan informasi sebelumnya dan kandidat informasi baru.

### 7.3 Kelebihan dan keterbatasan GRU

**Kelebihan:**

- Arsitektur lebih ringkas daripada LSTM standar.
- Tidak membutuhkan cell state terpisah.
- Dapat memiliki jumlah parameter lebih sedikit dan training lebih cepat pada beberapa kondisi.

**Keterbatasan:**

- Tetap memproses sequence secara berurutan.
- Tidak selalu lebih cepat atau lebih akurat pada setiap perangkat dan dataset.
- Pemilihan GRU atau LSTM sebaiknya didasarkan pada eksperimen, bukan asumsi bahwa satu model selalu lebih baik.

## 8. Contoh Kode GRU dengan PyTorch

```python
import torch
import torch.nn as nn

x = torch.randn(2, 4, 3)

gru = nn.GRU(
    input_size=3,
    hidden_size=5,
    batch_first=True
)

output, h_n = gru(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Final hidden state shape:", h_n.shape)
```

### Contoh output

```text
Input shape: torch.Size([2, 4, 3])
Output shape: torch.Size([2, 4, 5])
Final hidden state shape: torch.Size([1, 2, 5])
```

GRU menghasilkan `output` dan `h_n`, tetapi tidak menghasilkan `c_n` seperti LSTM.

## 9. Perbandingan RNN, LSTM, dan GRU

| Aspek | RNN sederhana | LSTM | GRU |
|---|---|---|---|
| Hidden state | Ya | Ya | Ya |
| Cell state terpisah | Tidak | Ya | Tidak |
| Mekanisme gate | Tidak seperti LSTM/GRU | Forget, input, output | Reset, update |
| Kompleksitas | Paling sederhana | Umumnya lebih kompleks | Umumnya di antara RNN dan LSTM |
| Ketergantungan panjang | Sering kesulitan | Dirancang untuk membantu | Dirancang untuk membantu |
| Pemrosesan berurutan | Ya | Ya | Ya |
| Pilihan terbaik untuk semua tugas? | Tidak | Tidak | Tidak |

## 10. Perbandingan Bentuk Tensor

Gunakan konfigurasi yang sama untuk ketiga model:

- `batch_size = 2`
- `sequence_length = 4`
- `input_size = 3`
- `hidden_size = 5`

| Model | `output` | `h_n` | State tambahan |
|---|---|---|---|
| RNN | `(2, 4, 5)` | `(1, 2, 5)` | Tidak ada |
| LSTM | `(2, 4, 5)` | `(1, 2, 5)` | `c_n`: `(1, 2, 5)` |
| GRU | `(2, 4, 5)` | `(1, 2, 5)` | Tidak ada |

Dimensi pertama `h_n` adalah `num_layers × num_directions`. Pada contoh ini nilainya 1 karena model memakai satu layer dan satu arah.

## 11. Contoh Model Klasifikasi Sequence

Contoh berikut menunjukkan bagaimana output recurrent network dapat digunakan untuk klasifikasi biner. Input masih berupa tensor acak, sehingga prediksi belum memiliki makna semantik dan model belum dilatih.

```python
import torch
import torch.nn as nn

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, h_n = self.gru(x)
        final_hidden = h_n[-1]  # (batch_size, hidden_size)
        logits = self.classifier(final_hidden)
        return logits

# 2 contoh, 4 time step, 3 fitur input
x = torch.randn(2, 4, 3)

model = GRUClassifier(input_size=3, hidden_size=5, num_classes=2)
logits = model(x)
predictions = torch.argmax(logits, dim=1)

print("Logits shape:", logits.shape)
print("Predictions shape:", predictions.shape)
print("Predicted classes:", predictions.tolist())
```

### Contoh output

```text
Logits shape: torch.Size([2, 2])
Predictions shape: torch.Size([2])
Predicted classes: [1, 0]
```

**Penting:** angka kelas pada contoh dapat berbeda setiap kali dijalankan karena bobot model diinisialisasi secara acak. Ini hanya demonstrasi alur dan shape, bukan hasil model terlatih.

### Memahami output

- `logits`: skor mentah untuk setiap kelas; belum berupa probabilitas.
- `predictions`: indeks kelas dengan skor tertinggi.
- Untuk mengubah logits menjadi probabilitas, gunakan `torch.softmax(logits, dim=1)`.
- Untuk training klasifikasi multikelas/biner dengan label integer, `nn.CrossEntropyLoss` dapat digunakan pada logits tanpa menerapkan softmax terlebih dahulu.

## 12. Gambaran Training Loop

Model belum belajar hanya dengan dibuat dan dipanggil. Training memerlukan label, fungsi loss, optimizer, dan proses pembaruan parameter.

Contoh ringkas satu langkah training:

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

model = GRUClassifier(input_size=3, hidden_size=5, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

x = torch.randn(2, 4, 3)
y = torch.tensor([1, 0])  # label untuk dua contoh

model.train()
optimizer.zero_grad()

logits = model(x)
loss = criterion(logits, y)

loss.backward()
optimizer.step()

print("Logits shape:", logits.shape)
print("Loss:", round(loss.item(), 4))
```

### Contoh output

```text
Logits shape: torch.Size([2, 2])
Loss: (nilai numerik bergantung pada hasil eksekusi)
```

Loss aktual bergantung pada tensor input dan parameter model. Satu langkah ini hanya demonstrasi mekanisme training; model membutuhkan banyak batch dan evaluasi pada data terpisah untuk menjadi model yang berguna.

## 13. Bagaimana Menerapkannya pada Teks?

Contoh input penelitian:

> “Tolong bantu kami, rumah terendam banjir.”

Sebelum teks bisa masuk ke RNN/LSTM/GRU, dibutuhkan beberapa tahap:

```text
Teks mentah
   ↓
Preprocessing yang sesuai
   ↓
Tokenization
   ↓
Vocabulary / token IDs
   ↓
Embedding
   ↓
RNN / LSTM / GRU
   ↓
Classification layer
   ↓
Emergency / Non-Emergency
```

Dalam implementasi nyata, token IDs biasanya diubah menjadi embedding menggunakan `nn.Embedding` atau embedding yang telah dilatih sebelumnya. Panjang sequence juga perlu ditangani dengan padding dan, bila perlu, masking atau packed sequences.

### Hal yang perlu diperhatikan

- Pisahkan data train, validation, dan test dengan benar.
- Bangun vocabulary hanya dari data training untuk menghindari kebocoran informasi.
- Tangani token di luar vocabulary.
- Tentukan panjang sequence dan strategi padding.
- Bandingkan model dengan baseline yang masuk akal, misalnya TF-IDF + SVM.
- Untuk deteksi bantuan darurat, evaluasi recall kelas Emergency, precision, F1-score, dan confusion matrix; jangan hanya melaporkan accuracy.

## 14. Hubungan dengan Attention dan Transformer

RNN, LSTM, dan GRU memproses token secara berurutan dan membawa hidden state dari satu langkah ke langkah berikutnya. Mekanisme ini berbeda dari self-attention pada Transformer, yang memungkinkan token membangun representasi dengan memperhatikan token lain dalam sequence melalui attention.

```text
RNN → LSTM → GRU
          ↓
   Pemahaman model sequence
          ↓
Attention Mechanism
          ↓
Transformer
          ↓
BERT
          ↓
IndoBERT
```

RNN/LSTM/GRU tetap berguna untuk memahami evolusi model NLP dan sebagai pembanding eksperimental. Namun, untuk target penelitian berbasis IndoBERT, Anda tidak harus membangun sistem RNN/LSTM/GRU produksi terlebih dahulu. Pahami konsep, jalankan contoh, lalu lanjut ke Attention dan Transformer.

## 15. Latihan

### Latihan 1 — Memahami tensor

Jalankan contoh RNN, LSTM, dan GRU dengan:

- `batch_size = 3`
- `sequence_length = 6`
- `input_size = 4`
- `hidden_size = 8`

Prediksi shape `output`, `h_n`, dan `c_n` sebelum menjalankan kode.

### Latihan 2 — Mengubah hidden size

Ubah `hidden_size` dari 5 menjadi 10. Amati dimensi `output` dan `h_n`.

### Latihan 3 — Membandingkan arsitektur

Buat tabel yang menjelaskan:

1. Apakah model mempunyai cell state terpisah?
2. Gate apa yang digunakan?
3. Apa output yang dihasilkan?
4. Apa kelebihan dan keterbatasannya?

### Latihan 4 — Klasifikasi teks

Setelah memahami tensor, buat pipeline sederhana untuk teks berlabel. Gunakan split train/validation/test, embedding, satu model recurrent, loss function, optimizer, dan evaluasi.

Jangan menggunakan tensor acak sebagai bukti keberhasilan klasifikasi teks. Gunakan data berlabel dan laporkan hasil evaluasi pada data yang tidak dipakai untuk training.

## 16. Checklist Penguasaan

- [ ] Saya dapat menjelaskan sequence, token, embedding, dan hidden state.
- [ ] Saya memahami alur hidden state pada RNN.
- [ ] Saya dapat menjelaskan vanishing gradient secara konseptual.
- [ ] Saya memahami cell state serta forget, input, dan output gate pada LSTM.
- [ ] Saya memahami reset dan update gate pada GRU.
- [ ] Saya dapat membaca shape output RNN, LSTM, dan GRU.
- [ ] Saya memahami perbedaan logits, probabilitas, dan prediksi kelas.
- [ ] Saya memahami bahwa model perlu dilatih dan dievaluasi sebelum digunakan.
- [ ] Saya siap melanjutkan ke Attention Mechanism.

## 17. Ringkasan

- **RNN** membawa hidden state dari satu time step ke time step berikutnya.
- **LSTM** menambahkan cell state dan tiga gate utama untuk mengatur aliran informasi.
- **GRU** menggunakan reset gate dan update gate dengan struktur yang lebih ringkas.
- Ketiga model memproses sequence secara berurutan.
- Contoh kode di materi ini terutama memperlihatkan struktur model dan bentuk tensor, bukan performa klasifikasi.
- Setelah konsep RNN → LSTM → GRU dipahami, langkah kurikulum berikutnya adalah **Attention Mechanism → Transformer → BERT → IndoBERT**.
