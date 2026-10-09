# Deep Learning untuk NLP: Dari Neural Network hingga RNN, LSTM, dan GRU

## Tujuan Pembelajaran

Materi ini merupakan bagian dari roadmap belajar Natural Language Processing (NLP), setelah mempelajari Fundamental NLP, Text Preprocessing, Text Representation, Machine Learning untuk NLP, dan Evaluasi NLP.

Setelah menyelesaikan materi ini, Anda diharapkan mampu:

1. Menjelaskan perbedaan Machine Learning klasik dan Deep Learning.
2. Memahami neuron, bobot, bias, fungsi aktivasi, loss function, dan optimizer.
3. Memahami forward propagation dan backpropagation secara konseptual.
4. Menggunakan tensor dan PyTorch untuk membangun neural network sederhana.
5. Memahami embedding sebagai representasi numerik untuk token.
6. Menjelaskan konsep sequence, RNN, LSTM, dan GRU.
7. Melatih model sederhana dan membaca output serta metriknya.
8. Memahami keterbatasan RNN/LSTM sebagai bekal mempelajari Attention, Transformer, BERT, dan IndoBERT.

> **Catatan:** Materi ini berfokus pada konsep Deep Learning yang diperlukan untuk memahami perkembangan NLP modern. Anda tidak harus menjadi ahli Deep Learning sebelum mempelajari Transformer atau melakukan fine-tuning IndoBERT.

## Prasyarat

Sebaiknya Anda sudah memahami:

- Variabel, fungsi, loop, list, dan dictionary Python.
- NumPy dan Pandas.
- Konsep fitur (`X`) dan target (`y`).
- `train_test_split`, klasifikasi, dan evaluasi model.
- Dasar vektor dan matriks.
- Konsep dasar text preprocessing dan TF-IDF.

## Peta Materi

1. Pengantar Deep Learning
2. Neural Network dan neuron buatan
3. Weight, bias, dan fungsi aktivasi
4. Forward propagation
5. Loss function
6. Backpropagation dan optimizer
7. Overfitting, underfitting, dan regularisasi
8. Pengenalan PyTorch dan tensor
9. Neural network dengan PyTorch
10. Embedding untuk teks
11. Sequence learning
12. Recurrent Neural Network (RNN)
13. Vanishing gradient
14. Long Short-Term Memory (LSTM)
15. Gated Recurrent Unit (GRU)
16. Eksperimen klasifikasi teks sederhana
17. Evaluasi dan analisis hasil
18. Hubungan Deep Learning dengan Attention dan Transformer
19. Checklist penguasaan dan latihan

## 1. Pengantar Deep Learning

### 1.1 Apa itu Deep Learning?

Deep Learning adalah bagian dari Machine Learning yang menggunakan neural network dengan beberapa lapisan untuk mempelajari pola dari data. Model mempelajari parameter dari contoh data melalui proses training.

Dalam NLP, Deep Learning dapat digunakan untuk:

- Klasifikasi teks.
- Analisis sentimen.
- Deteksi spam.
- Named Entity Recognition (NER).
- Pemodelan bahasa.
- Klasifikasi permintaan bantuan darurat dari teks media sosial.

Contoh alur klasifikasi teks:

```text
Teks
  ↓
Representasi numerik
  ↓
Neural Network
  ↓
Probabilitas kelas
  ↓
Label prediksi
```

### 1.2 Machine Learning klasik vs Deep Learning

| Aspek | ML klasik untuk teks | Deep Learning untuk teks |
|---|---|---|
| Representasi umum | Bag of Words atau TF-IDF | Embedding yang dapat dipelajari atau embedding pralatih |
| Fitur | Sering direkayasa atau ditentukan melalui vectorizer | Dapat dipelajari bersama parameter model |
| Contoh model | Logistic Regression, Naive Bayes, SVM | MLP, RNN, LSTM, GRU, Transformer |
| Kebutuhan komputasi | Umumnya lebih ringan | Dapat lebih tinggi |
| Data | Dapat bekerja baik pada dataset kecil hingga menengah | Model kompleks sering membutuhkan data/komputasi lebih besar |
| Interpretasi | Beberapa model relatif mudah dianalisis | Penjelasan keputusan dapat lebih menantang |

Deep Learning tidak selalu lebih baik. Untuk penelitian, model Deep Learning harus dibandingkan dengan baseline yang wajar, misalnya TF-IDF + Logistic Regression atau TF-IDF + SVM.

### 1.3 Contoh penerapan pada penelitian kebencanaan

Misalnya, sistem menerima teks:

> "Tolong bantu kami, rumah sudah terendam banjir."

Model klasifikasi dapat menghasilkan:

```text
Prediksi: Emergency
Probabilitas model: 0.91
```

Angka probabilitas di atas hanya contoh ilustratif, bukan hasil eksperimen nyata. Prediksi model juga bukan pengganti verifikasi manusia dalam keputusan tanggap darurat.

## 2. Neural Network

### 2.1 Neuron buatan

Neuron buatan menerima sejumlah input, mengalikannya dengan bobot, menambahkan bias, lalu menerapkan fungsi aktivasi.

Rumus:

\[
z = \sum_{i=1}^{n} w_i x_i + b
\]

\[
a = f(z)
\]

Keterangan:

- \(x_i\): input ke-\(i\).
- \(w_i\): bobot untuk input ke-\(i\).
- \(b\): bias.
- \(z\): nilai sebelum aktivasi.
- \(f\): fungsi aktivasi.
- \(a\): keluaran neuron.

Bobot dan bias adalah parameter yang dipelajari selama training.

### 2.2 Contoh perhitungan dengan Python

Kode berikut menghitung satu neuron sederhana tanpa library Deep Learning.

```python
x1, x2 = 0.7, 0.2
w1, w2 = 0.8, -0.4
bias = 0.1

z = (x1 * w1) + (x2 * w2) + bias

print(f"Nilai z: {z:.2f}")
```

Output:

```text
Nilai z: 0.58
```

Nilai `z` tersebut belum melewati fungsi aktivasi.

### 2.3 Fungsi aktivasi

Fungsi aktivasi menambahkan sifat non-linear agar jaringan dapat mempelajari pola yang lebih kompleks.

| Fungsi | Bentuk sederhana | Kegunaan umum |
|---|---|---|
| Sigmoid | Menghasilkan nilai antara 0 dan 1 | Output klasifikasi biner tertentu |
| Tanh | Menghasilkan nilai antara -1 dan 1 | Ditemukan pada beberapa arsitektur recurrent |
| ReLU | \(\max(0,z)\) | Umum pada hidden layer |
| Softmax | Mengubah skor menjadi distribusi probabilitas | Output klasifikasi multikelas |

Contoh implementasi:

```python
import numpy as np

z = np.array([-2.0, 0.0, 2.0])

relu = np.maximum(0, z)
sigmoid = 1 / (1 + np.exp(-z))

print("ReLU:", relu)
print("Sigmoid:", np.round(sigmoid, 3))
```

Output:

```text
ReLU: [0. 0. 2.]
Sigmoid: [0.119 0.5   0.881]
```

Untuk klasifikasi biner, sigmoid sering digunakan pada output satu neuron. Untuk klasifikasi multikelas dengan label yang saling eksklusif, model biasanya menghasilkan logits untuk setiap kelas dan menggunakan cross-entropy loss; probabilitas dapat diperoleh dengan softmax.

## 3. Lapisan Neural Network

### 3.1 Jenis lapisan

Jaringan neural sederhana dapat terdiri dari:

```text
Input Layer
    ↓
Hidden Layer
    ↓
Output Layer
```

- **Input layer:** menerima fitur.
- **Hidden layer:** mempelajari kombinasi pola.
- **Output layer:** menghasilkan skor atau prediksi.

Dalam NLP, input bisa berupa vektor TF-IDF atau representasi embedding.

### 3.2 Istilah penting

- **Parameter:** bobot dan bias yang dipelajari model.
- **Hyperparameter:** pengaturan training seperti learning rate, batch size, jumlah epoch, dan jumlah hidden unit.
- **Epoch:** satu kali model melewati seluruh data training.
- **Batch:** sebagian data yang diproses dalam satu langkah.
- **Step/iteration:** satu pembaruan parameter, biasanya setelah satu batch diproses.

## 4. Forward Propagation

Forward propagation adalah proses mengalirkan input melalui jaringan untuk menghasilkan prediksi.

```text
Input
  ↓
Linear transformation
  ↓
Activation
  ↓
Hidden layer
  ↓
Output logits
  ↓
Loss
```

Pada satu lapisan linear:

\[
z = Wx + b
\]

Kemudian:

\[
a = f(z)
\]

Proses ini dilakukan dari input sampai output. Pada tahap ini, model belum memperbarui bobot.

## 5. Loss Function

Loss function mengukur seberapa jauh prediksi model dari target yang benar. Training berusaha mengurangi loss.

### 5.1 Mean Squared Error (MSE)

MSE lazim digunakan pada regresi:

\[
MSE = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
\]

### 5.2 Binary Cross-Entropy

Binary cross-entropy digunakan untuk klasifikasi biner dengan probabilitas prediksi. Dalam PyTorch, `BCEWithLogitsLoss` lebih disarankan ketika model mengeluarkan logits karena menggabungkan sigmoid dan loss secara numerik stabil.

### 5.3 Cross-Entropy

Untuk klasifikasi multikelas dengan satu kelas benar per contoh, PyTorch menyediakan `CrossEntropyLoss`. Fungsi ini menerima **logits**, bukan probabilitas softmax yang sudah dihitung.

Contoh konseptual:

```text
Label sebenarnya: Emergency
Prediksi model:   [Non-Emergency: 0.20, Emergency: 0.80]
```

Prediksi dengan probabilitas tinggi pada kelas yang benar akan menghasilkan loss yang relatif lebih kecil daripada prediksi yang sangat yakin pada kelas yang salah.

## 6. Backpropagation dan Optimizer

### 6.1 Backpropagation

Backpropagation menghitung gradien loss terhadap parameter menggunakan aturan rantai (chain rule). Gradien menunjukkan bagaimana parameter berkontribusi terhadap perubahan loss.

Alur training:

```text
Input dan label
      ↓
Forward propagation
      ↓
Hitung loss
      ↓
Backpropagation (hitung gradien)
      ↓
Optimizer memperbarui parameter
      ↓
Ulangi untuk batch berikutnya
```

### 6.2 Gradient descent

Secara sederhana, pembaruan parameter dapat ditulis:

\[
\theta_{t+1} = \theta_t - \eta \nabla_\theta L
\]

Keterangan:

- \(\theta\): parameter model.
- \(\eta\): learning rate.
- \(\nabla_\theta L\): gradien loss terhadap parameter.

Learning rate mengatur ukuran langkah pembaruan. Jika terlalu besar, training dapat tidak stabil; jika terlalu kecil, training dapat berjalan lambat.

### 6.3 Optimizer yang umum

- **SGD:** stochastic gradient descent.
- **Adam:** optimizer adaptif yang umum digunakan.
- **AdamW:** variasi Adam dengan weight decay yang dipisahkan dari pembaruan gradien; umum pada training Transformer.

Untuk latihan awal, gunakan Adam. Saat masuk IndoBERT, Anda akan sering menemukan AdamW.

## 7. Overfitting, Underfitting, dan Regularisasi

### 7.1 Underfitting

Model terlalu sederhana atau belum belajar cukup baik sehingga performa training dan validasi sama-sama buruk.

### 7.2 Overfitting

Model sangat baik pada data training, tetapi performanya menurun pada data yang belum pernah dilihat.

Contoh ilustratif:

| Epoch | Training accuracy | Validation accuracy |
|---:|---:|---:|
| 1 | 0.70 | 0.68 |
| 5 | 0.88 | 0.82 |
| 10 | 0.99 | 0.76 |

Tabel tersebut hanya ilustrasi. Pola ini dapat mengindikasikan overfitting ketika performa training terus meningkat tetapi performa validasi menurun.

### 7.3 Teknik pengendalian

- **Early stopping:** menghentikan training ketika metrik validasi tidak lagi membaik.
- **Dropout:** menonaktifkan sebagian unit secara acak selama training.
- **Weight decay:** memberi penalti pada besarnya parameter.
- **Data augmentation:** membuat variasi data yang tetap mempertahankan label, jika sesuai dengan tugas.
- **Data yang representatif:** memastikan data validasi/test mencerminkan kondisi penggunaan.
- **Split yang benar:** hindari kebocoran data antara training dan test.

Dalam penelitian, jangan memilih model berdasarkan hasil test berulang kali. Gunakan validation set untuk pemilihan model dan simpan test set untuk evaluasi akhir.

## 8. Pengenalan PyTorch

### 8.1 Instalasi

Ikuti instruksi instalasi resmi PyTorch yang sesuai dengan sistem operasi dan dukungan CPU/GPU Anda. Untuk lingkungan Python yang sudah aktif, instalasi CPU sederhana dapat dilakukan dengan:

```bash
pip install torch
```

Untuk GPU, periksa panduan resmi PyTorch agar versi paket cocok dengan lingkungan CUDA yang digunakan.

### 8.2 Tensor

Tensor adalah struktur data utama PyTorch, mirip array NumPy tetapi mendukung automatic differentiation dan komputasi GPU.

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

print(x)
print("Shape:", x.shape)
print("Jumlah seluruh elemen:", x.sum().item())
```

Output:

```text
tensor([[1., 2.],
        [3., 4.]])
Shape: torch.Size([2, 2])
Jumlah seluruh elemen: 10.0
```

### 8.3 Automatic differentiation

PyTorch dapat menghitung gradien secara otomatis.

```python
import torch

w = torch.tensor(2.0, requires_grad=True)
loss = (w - 5) ** 2

loss.backward()

print("Loss:", loss.item())
print("Gradien:", w.grad.item())
```

Output:

```text
Loss: 9.0
Gradien: -6.0
```

Penjelasan: \(L=(w-5)^2\), sehingga turunannya terhadap \(w\) adalah \(2(w-5)\). Saat \(w=2\), gradiennya adalah \(-6\).

## 9. Membangun Neural Network Sederhana dengan PyTorch

Contoh berikut melatih neural network untuk mempelajari pola XOR. XOR adalah masalah mainan yang tidak dapat diselesaikan dengan satu pemisah linear, sehingga berguna untuk memperlihatkan manfaat hidden layer non-linear.

```python
import torch
from torch import nn

torch.manual_seed(42)

# Empat kombinasi input XOR
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
])

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

for epoch in range(2000):
    model.train()

    logits = model(X)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    probabilities = torch.sigmoid(model(X))
    predictions = (probabilities >= 0.5).int()

print("Probabilitas:")
print(probabilities.round(decimals=3))
print("Prediksi:")
print(predictions.squeeze().tolist())
print("Label aktual:")
print(y.squeeze().int().tolist())
```

Output yang diharapkan setelah training:

```text
Probabilitas:
tensor([[mendekati 0],
        [mendekati 1],
        [mendekati 1],
        [mendekati 0]])

Prediksi:
[0, 1, 1, 0]
Label aktual:
[0, 1, 1, 0]
```

Nilai probabilitas persis dapat sedikit berbeda bergantung pada versi perangkat lunak dan lingkungan. Tujuan latihan ini adalah memahami komponen model, loss, gradien, dan optimizer—bukan menggunakan XOR sebagai benchmark penelitian.

### 9.1 Memahami bagian kode

- `nn.Linear(2, 8)`: mengubah dua fitur menjadi delapan unit.
- `nn.ReLU()`: fungsi aktivasi non-linear.
- `nn.Linear(8, 1)`: menghasilkan satu logit.
- `BCEWithLogitsLoss()`: menghitung loss klasifikasi biner dari logits.
- `optimizer.zero_grad()`: menghapus gradien dari langkah sebelumnya.
- `loss.backward()`: menghitung gradien.
- `optimizer.step()`: memperbarui parameter.
- `torch.no_grad()`: menonaktifkan pencatatan gradien saat inferensi.

## 10. Embedding untuk Teks

### 10.1 Mengapa embedding diperlukan?

Model neural network membutuhkan input numerik. Untuk teks, setiap token dapat diwakili oleh vektor.

Berbeda dari one-hot encoding yang sangat jarang (sparse) dan berdimensi sebesar vocabulary, embedding memetakan token ke vektor berdimensi lebih kecil yang nilainya dapat dipelajari.

Contoh konseptual:

```text
"banjir"     → [ 0.12, -0.31, 0.77, ...]
"bantuan"    → [-0.08,  0.62, 0.15, ...]
```

Angka di atas hanya ilustrasi, bukan embedding hasil model nyata.

### 10.2 `nn.Embedding` di PyTorch

```python
import torch
from torch import nn

# Misalnya vocabulary terdiri dari 6 token
embedding = nn.Embedding(num_embeddings=6, embedding_dim=4)

# ID token untuk satu contoh
token_ids = torch.tensor([1, 3, 4])

vectors = embedding(token_ids)

print("Shape token IDs:", token_ids.shape)
print("Shape embedding:", vectors.shape)
```

Output:

```text
Shape token IDs: torch.Size([3])
Shape embedding: torch.Size([3, 4])
```

Penjelasan:

- Ada 3 token pada input.
- Setiap token dipetakan ke vektor berdimensi 4.
- Nilai embedding diinisialisasi dan dipelajari selama training, kecuali Anda menggunakan embedding pralatih atau membekukan parameternya.

### 10.3 Embedding vs TF-IDF

| Aspek | TF-IDF | Embedding yang dipelajari |
|---|---|---|
| Representasi | Bobot kata dalam dokumen/korpus | Vektor padat untuk token |
| Konteks | Umumnya tidak menangkap konteks urutan secara langsung | Bergantung arsitektur; embedding dasar sendiri belum tentu kontekstual |
| Dimensi | Umumnya terkait ukuran vocabulary | Dimensi embedding ditentukan |
| Pembelajaran | Statistik korpus | Parameter dapat dipelajari dari tugas atau pralatih |
| Contoh | TF-IDF + SVM | Embedding + LSTM atau Transformer |

Perlu dibedakan: embedding Word2Vec biasanya memiliki satu vektor tetap per kata, sedangkan representasi kontekstual BERT berubah menurut konteks kalimat.

## 11. Sequence Learning

Bahasa merupakan urutan token. Urutan dapat memengaruhi makna.

Bandingkan:

- "Petugas menolong warga."
- "Warga menolong petugas."

Kata-katanya mirip, tetapi peran subjek dan objek berbeda.

Representasi bag-of-words mengabaikan urutan token. Model sequence seperti RNN, LSTM, dan GRU dirancang untuk memproses urutan.

Contoh sequence token:

```text
"Tolong bantu kami"
       ↓
["Tolong", "bantu", "kami"]
       ↓
[token_id_1, token_id_2, token_id_3]
       ↓
[embedding_1, embedding_2, embedding_3]
```

## 12. Recurrent Neural Network (RNN)

### 12.1 Konsep RNN

RNN memproses urutan token langkah demi langkah dan membawa hidden state dari satu langkah ke langkah berikutnya.

Secara konseptual:

\[
h_t = f(W_x x_t + W_h h_{t-1} + b)
\]

Keterangan:

- \(x_t\): representasi token pada waktu ke-\(t\).
- \(h_{t-1}\): hidden state sebelumnya.
- \(h_t\): hidden state baru.
- \(W_x, W_h\): bobot yang dipelajari.
- \(f\): fungsi aktivasi.

Alur:

```text
Token 1 → RNN → hidden state 1
                      ↓
Token 2 → RNN → hidden state 2
                      ↓
Token 3 → RNN → hidden state 3
                      ↓
                  Prediksi
```

Bobot yang sama digunakan pada langkah-langkah sequence. Hidden state berfungsi membawa sebagian informasi dari token sebelumnya.

### 12.2 Keterbatasan RNN

- Pemrosesan sequence bersifat berurutan sehingga sulit diparalelkan sepenuhnya sepanjang waktu.
- Informasi dari token yang jauh dapat sulit dipertahankan.
- Training sequence panjang dapat mengalami vanishing gradient atau exploding gradient.

Keterbatasan ini menjadi salah satu motivasi pengembangan LSTM, GRU, dan kemudian Transformer.

## 13. Vanishing Gradient dan Exploding Gradient

### 13.1 Vanishing gradient

Gradien menjadi sangat kecil saat dipropagasikan melalui banyak langkah. Akibatnya, parameter yang berkaitan dengan informasi lama mungkin diperbarui sangat sedikit.

Dampak potensial:

- Sulit mempelajari ketergantungan jangka panjang.
- Informasi dari bagian awal sequence sulit dimanfaatkan.

### 13.2 Exploding gradient

Gradien menjadi sangat besar sehingga pembaruan parameter tidak stabil.

Salah satu teknik yang dapat membantu adalah gradient clipping. PyTorch menyediakan `torch.nn.utils.clip_grad_norm_`, tetapi teknik ini tidak menghilangkan seluruh masalah pembelajaran sequence.

## 14. Long Short-Term Memory (LSTM)

### 14.1 Apa itu LSTM?

LSTM adalah jenis RNN yang menggunakan mekanisme gerbang (gates) dan cell state untuk membantu mengatur informasi yang dipertahankan atau dibuang.

Tiga gerbang utama:

- **Forget gate:** menentukan informasi yang dipertahankan dari cell state sebelumnya.
- **Input gate:** mengatur informasi baru yang ditambahkan.
- **Output gate:** mengatur informasi yang digunakan untuk hidden state saat ini.

Diagram konseptual:

```text
Input token ───────────────┐
                           ↓
Hidden state sebelumnya → LSTM
Cell state sebelumnya  → LSTM
                           ↓
             Hidden state dan cell state baru
```

### 14.2 Intuisi contoh

Kalimat:

> "Air naik dengan cepat. Warga membutuhkan perahu untuk evakuasi."

Model ideal perlu mempertahankan informasi dari bagian awal kalimat untuk membantu memahami kebutuhan evakuasi di bagian berikutnya. LSTM dirancang agar informasi penting dapat dipertahankan lebih baik daripada RNN sederhana, meskipun tidak menjamin seluruh konteks selalu diingat.

### 14.3 LSTM di PyTorch

```python
import torch
from torch import nn

lstm = nn.LSTM(
    input_size=8,
    hidden_size=16,
    num_layers=1,
    batch_first=True
)

# Batch 2, sequence length 5, embedding dimension 8
x = torch.randn(2, 5, 8)

output, (h_n, c_n) = lstm(x)

print("Input:", x.shape)
print("Output seluruh timestep:", output.shape)
print("Hidden state akhir:", h_n.shape)
print("Cell state akhir:", c_n.shape)
```

Output:

```text
Input: torch.Size([2, 5, 8])
Output seluruh timestep: torch.Size([2, 5, 16])
Hidden state akhir: torch.Size([1, 2, 16])
Cell state akhir: torch.Size([1, 2, 16])
```

Penjelasan:

- `batch_first=True` berarti bentuk input adalah `(batch, sequence, feature)`.
- `input_size=8` berarti setiap token memiliki vektor 8 dimensi.
- `hidden_size=16` berarti hidden state berdimensi 16.
- `output` menyimpan hidden state pada seluruh timestep.
- `h_n` adalah hidden state akhir tiap layer.
- `c_n` adalah cell state akhir tiap layer.

## 15. Gated Recurrent Unit (GRU)

### 15.1 Apa itu GRU?

GRU juga merupakan varian RNN yang menggunakan gerbang untuk mengatur aliran informasi. Dibandingkan LSTM, GRU menggabungkan mekanisme tertentu dan tidak memiliki cell state terpisah.

Secara umum, GRU memiliki:

- **Update gate:** mengatur seberapa banyak informasi sebelumnya dipertahankan.
- **Reset gate:** mengatur bagaimana informasi sebelumnya digunakan saat membentuk kandidat state baru.

### 15.2 GRU di PyTorch

```python
import torch
from torch import nn

gru = nn.GRU(
    input_size=8,
    hidden_size=16,
    num_layers=1,
    batch_first=True
)

x = torch.randn(2, 5, 8)

output, h_n = gru(x)

print("Input:", x.shape)
print("Output seluruh timestep:", output.shape)
print("Hidden state akhir:", h_n.shape)
```

Output:

```text
Input: torch.Size([2, 5, 8])
Output seluruh timestep: torch.Size([2, 5, 16])
Hidden state akhir: torch.Size([1, 2, 16])
```

### 15.3 Perbandingan RNN, LSTM, dan GRU

| Aspek | RNN sederhana | LSTM | GRU |
|---|---|---|---|
| Mekanisme gerbang | Tidak memiliki gerbang LSTM/GRU | Forget, input, output gates | Reset dan update gates |
| State | Hidden state | Hidden state + cell state | Hidden state |
| Ketergantungan panjang | Sering sulit pada sequence panjang | Dirancang membantu mengatasi masalah ini | Dirancang membantu mengatasi masalah ini |
| Kompleksitas | Paling sederhana | Lebih banyak parameter | Sering lebih sederhana daripada LSTM |
| Pilihan terbaik | Berguna untuk memahami konsep | Berguna untuk sequence learning | Alternatif yang efisien untuk diuji |

Tidak ada model yang selalu terbaik. Performa perlu diuji pada dataset, preprocessing, dan pengaturan eksperimen yang sama.

## 16. Mini Project: Klasifikasi Teks dengan Embedding + LSTM

Bagian ini memperlihatkan bentuk pipeline, bukan benchmark penelitian. Data sangat kecil di bawah hanya untuk demonstrasi sintaks. Model tidak akan memiliki kemampuan generalisasi yang dapat diandalkan dari contoh ini.

### 16.1 Data contoh

```python
texts = [
    "tolong butuh bantuan",
    "rumah terendam banjir",
    "mohon kirim perahu",
    "saya sedang makan",
    "film itu sangat bagus",
    "besok saya pergi bekerja",
]

labels = [1, 1, 1, 0, 0, 0]
```

Dalam contoh ini:

- `1` = emergency.
- `0` = non-emergency.

Dalam penelitian sungguhan, definisi kelas dan aturan anotasi harus dirumuskan secara eksplisit. Frasa yang menyebut bencana belum tentu merupakan permintaan bantuan darurat.

### 16.2 Representasi token sederhana

Untuk memudahkan demonstrasi, contoh ini menggunakan pemisahan berdasarkan spasi dan vocabulary buatan sendiri. Untuk eksperimen yang benar, vocabulary hanya boleh dibangun dari data training dan pipeline harus menangani token tak dikenal serta padding dengan benar.

```python
from collections import Counter

tokens_per_text = [text.lower().split() for text in texts]
vocab = {"<PAD>": 0, "<UNK>": 1}

for token in sorted(set(token for row in tokens_per_text for token in row)):
    vocab[token] = len(vocab)

print("Jumlah token unik termasuk token khusus:", len(vocab))
print("Contoh vocabulary:", list(vocab.items())[:8])
```

Output: jumlah vocabulary mengikuti teks contoh yang diberikan. Entri awalnya akan dimulai dari `('<PAD>', 0)` dan `('<UNK>', 1)`.

### 16.3 Model konseptual

Arsitektur yang akan dibuat:

```text
Token IDs
   ↓
Embedding
   ↓
LSTM
   ↓
Linear layer
   ↓
Logit
   ↓
Binary classification
```

Untuk mengubah teks menjadi token ID, mem-padding sequence, membuat `Dataset`/`DataLoader`, dan melatih model secara benar, diperlukan pemisahan training/validation/test. Jangan menilai model dari data yang sama dengan data training.

> **Latihan lanjutan:** implementasikan pipeline ini setelah Anda memahami `Dataset`, `DataLoader`, padding sequence, masking, dan evaluasi klasifikasi. Untuk penelitian IndoBERT, Anda nantinya akan menggunakan tokenizer bawaan model alih-alih vocabulary berbasis spasi.

## 17. Praktik Training yang Baik

### 17.1 Pembagian data

Pisahkan data menjadi:

- **Training set:** memperbarui parameter model.
- **Validation set:** memilih hyperparameter dan memantau overfitting.
- **Test set:** evaluasi akhir pada data yang tidak digunakan saat pengembangan.

Untuk data media sosial, pertimbangkan duplikasi tweet, retweet, kemiripan teks, serta pemisahan berdasarkan waktu atau kejadian bencana jika relevan. Pembagian acak biasa dapat menghasilkan estimasi yang terlalu optimistis jika teks serupa bocor ke beberapa subset.

### 17.2 Catat eksperimen

Minimal catat:

- Nama model dan konfigurasi.
- Seed random.
- Dataset dan aturan split.
- Jumlah contoh setiap kelas.
- Optimizer dan learning rate.
- Batch size dan epoch.
- Training loss dan validation loss.
- Precision, recall, F1, dan confusion matrix.
- Waktu training serta perangkat yang digunakan.

### 17.3 Jangan hanya mengandalkan accuracy

Dalam deteksi permintaan bantuan darurat, false negative perlu diperhatikan: laporan yang sebenarnya darurat tetapi diprediksi bukan darurat. Evaluasi recall kelas emergency dan precision kelas emergency bersama metrik lain. Pilihan ambang keputusan juga dapat memengaruhi trade-off tersebut.

## 18. Hubungan Deep Learning dengan Attention dan Transformer

RNN, LSTM, dan GRU memproses token secara berurutan. Transformer menggunakan attention untuk menghubungkan representasi token dengan token lain dalam sequence tanpa harus melewati state recurrent satu per satu.

Perkembangan konsep:

```text
Bag of Words / TF-IDF
        ↓
Word Embedding
        ↓
RNN → LSTM / GRU
        ↓
Attention Mechanism
        ↓
Transformer
        ↓
BERT
        ↓
IndoBERT
```

Anda tidak harus menguasai semua variasi RNN sebelum mempelajari Transformer. Fokuskan pemahaman pada:

1. Mengapa teks perlu direpresentasikan sebagai angka.
2. Bagaimana neural network mempelajari parameter.
3. Mengapa urutan token penting.
4. Apa keterbatasan RNN untuk konteks panjang.
5. Mengapa attention membantu membangun representasi kontekstual.

## 19. Checklist Penguasaan

Gunakan daftar berikut untuk menilai kesiapan Anda.

- [ ] Dapat menjelaskan neuron, weight, bias, dan activation function.
- [ ] Dapat membedakan parameter dan hyperparameter.
- [ ] Memahami forward propagation dan backpropagation.
- [ ] Dapat menjelaskan loss function dan optimizer.
- [ ] Dapat menggunakan tensor PyTorch.
- [ ] Dapat membangun dan melatih neural network sederhana.
- [ ] Memahami perbedaan embedding dan TF-IDF.
- [ ] Dapat menjelaskan sequence dan hidden state.
- [ ] Dapat membandingkan RNN, LSTM, dan GRU.
- [ ] Memahami vanishing gradient secara konseptual.
- [ ] Memahami overfitting dan penggunaan validation set.
- [ ] Siap melanjutkan ke Attention Mechanism dan Transformer.

## 20. Latihan

### Latihan 1 — Konsep

Jawab dengan kata-kata sendiri:

1. Apa perbedaan Machine Learning klasik dan Deep Learning?
2. Apa fungsi weight dan bias?
3. Apa perbedaan parameter dan hyperparameter?
4. Mengapa loss perlu dihitung?
5. Apa fungsi backpropagation?
6. Mengapa learning rate penting?
7. Apa perbedaan training set, validation set, dan test set?
8. Mengapa embedding berbeda dari TF-IDF?
9. Apa fungsi hidden state pada RNN?
10. Apa perbedaan utama LSTM dan GRU?

### Latihan 2 — Eksperimen PyTorch

Ubah contoh XOR di bagian 9:

1. Ubah learning rate dari `0.03` menjadi `0.01`.
2. Ubah hidden unit dari `8` menjadi `16`.
3. Bandingkan hasil prediksi dan loss.
4. Jalankan eksperimen beberapa kali dengan seed berbeda.
5. Catat apakah hasilnya konsisten.

Jangan menyimpulkan bahwa satu konfigurasi lebih baik hanya dari satu kali percobaan kecil.

### Latihan 3 — Analisis untuk NLP

Ambil 20 kalimat bahasa Indonesia dan jelaskan:

1. Bagaimana teks diubah menjadi token.
2. Bagaimana token diubah menjadi ID.
3. Bagaimana embedding menghasilkan vektor.
4. Bagaimana LSTM memproses sequence.
5. Bagaimana layer output menghasilkan prediksi kelas.

Gunakan data sintetis atau dataset yang izinnya sesuai. Jangan menganggap contoh kecil ini cukup untuk menyimpulkan performa sistem darurat di dunia nyata.

## 21. Ringkasan

Deep Learning mempelajari pola melalui neural network dan parameter yang diperbarui berdasarkan loss. PyTorch membantu membangun model, menghitung gradien, dan menjalankan training. Dalam NLP, embedding mengubah token menjadi vektor; RNN, LSTM, dan GRU memproses urutan token dengan mekanisme recurrent.

Untuk mencapai IndoBERT, urutan belajar yang disarankan adalah:

```text
Neural Network
    ↓
PyTorch
    ↓
Embedding
    ↓
RNN / LSTM / GRU
    ↓
Attention Mechanism
    ↓
Transformer
    ↓
BERT
    ↓
IndoBERT Fine-Tuning
```

Tahap berikutnya setelah materi ini adalah **Attention Mechanism**, lalu **Transformer**. Ketika memulai penelitian, bandingkan model Deep Learning dengan baseline TF-IDF + Logistic Regression/SVM dan lakukan evaluasi pada data uji yang benar-benar terpisah.
