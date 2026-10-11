# IndoBERT untuk Klasifikasi Teks Bahasa Indonesia

> **Level:** Menengah  
> **Prasyarat:** Python, dasar NLP, preprocessing, TF-IDF, evaluasi klasifikasi, serta konsep Transformer dan BERT  
> **Tujuan:** Memahami IndoBERT dan menerapkannya untuk klasifikasi teks bahasa Indonesia, sebagai dasar penelitian deteksi tweet permintaan bantuan darurat saat bencana.

## Daftar Isi

1. [Tujuan Pembelajaran](#1-tujuan-pembelajaran)
2. [Apa Itu IndoBERT?](#2-apa-itu-indobert)
3. [BERT dan IndoBERT](#3-bert-dan-indobert)
4. [Cara Kerja IndoBERT](#4-cara-kerja-indobert)
5. [Tokenizer IndoBERT](#5-tokenizer-indobert)
6. [Menyiapkan Environment](#6-menyiapkan-environment)
7. [Memuat Tokenizer dan Model](#7-memuat-tokenizer-dan-model)
8. [Memahami Hasil Tokenisasi](#8-memahami-hasil-tokenisasi)
9. [Menghasilkan Representasi Teks](#9-menghasilkan-representasi-teks)
10. [Fine-Tuning untuk Klasifikasi Teks](#10-fine-tuning-untuk-klasifikasi-teks)
11. [Evaluasi Model](#11-evaluasi-model)
12. [Membuat Prediksi Teks Baru](#12-membuat-prediksi-teks-baru)
13. [Preprocessing untuk IndoBERT](#13-preprocessing-untuk-indobert)
14. [IndoBERT dalam Penelitian Bantuan Darurat](#14-indobert-dalam-penelitian-bantuan-darurat)
15. [Kesalahan yang Sering Terjadi](#15-kesalahan-yang-sering-terjadi)
16. [Latihan](#16-latihan)
17. [Mini Project](#17-mini-project)
18. [Checkpoint](#18-checkpoint)
19. [Referensi](#19-referensi)

---

## 1. Tujuan Pembelajaran

Setelah menyelesaikan materi ini, Anda diharapkan mampu:

- Menjelaskan pengertian dan kegunaan IndoBERT.
- Memahami hubungan BERT, Transformer Encoder, dan IndoBERT.
- Menggunakan tokenizer IndoBERT.
- Memahami `input_ids`, `attention_mask`, padding, dan truncation.
- Memuat model pretrained melalui Hugging Face Transformers.
- Melakukan fine-tuning untuk klasifikasi teks.
- Mengevaluasi hasil menggunakan precision, recall, dan F1-score.
- Menjelaskan bagaimana IndoBERT dapat digunakan dalam penelitian deteksi permintaan bantuan darurat.

## 2. Apa Itu IndoBERT?

**IndoBERT** adalah keluarga model bahasa berbasis arsitektur BERT yang dilatih untuk memahami teks bahasa Indonesia. IndoBERT menghasilkan representasi kata atau subword yang dipengaruhi oleh konteks kalimatnya.

Contoh kalimat:

- `Saya melihat ular di jalan.`
- `Ia mengular di antrean panjang.`

Kata yang sama atau mirip dapat mempunyai makna berbeda bergantung pada konteks. Model kontekstual seperti BERT berusaha merepresentasikan teks dengan mempertimbangkan konteks tersebut.

### 2.1 Kegunaan IndoBERT

IndoBERT dapat digunakan sebagai dasar untuk berbagai tugas NLP, antara lain:

- Klasifikasi teks
- Analisis sentimen
- Klasifikasi berita
- Pengenalan entitas bernama (Named Entity Recognition), dengan model atau head yang sesuai
- Deteksi topik atau kategori dokumen
- Ekstraksi informasi, dengan penyesuaian tugas

IndoBERT **bukan otomatis sebuah sistem klasifikasi yang sudah sesuai dengan kebutuhan kita**. Untuk tugas tertentu, model biasanya perlu ditambahkan classification head dan di-fine-tune menggunakan dataset berlabel yang relevan.

### 2.2 IndoBERT bukan chatbot generatif

IndoBERT dasar adalah model encoder untuk representasi dan pemahaman teks. Model ini tidak dirancang sebagai chatbot generatif seperti model decoder-only. Untuk klasifikasi, IndoBERT dapat dipasangkan dengan classification head.

## 3. BERT dan IndoBERT

BERT merupakan singkatan dari *Bidirectional Encoder Representations from Transformers*. BERT menggunakan blok Transformer Encoder untuk mempelajari representasi teks dengan konteks dari kedua arah.

IndoBERT menerapkan pendekatan BERT pada bahasa Indonesia dengan proses pelatihan dan data yang ditujukan untuk bahasa Indonesia.

| Aspek | BERT umum | IndoBERT |
| --- | --- | --- |
| Arsitektur dasar | Transformer Encoder | Transformer Encoder berbasis BERT |
| Bahasa | Bergantung pada model yang dipilih; model BERT asli berfokus pada bahasa Inggris | Berfokus pada bahasa Indonesia |
| Tokenizer | Sesuai model | Sesuai checkpoint IndoBERT |
| Penggunaan | Bergantung pada data dan tugas | Cocok dipertimbangkan untuk tugas teks bahasa Indonesia |
| Kebutuhan fine-tuning | Umumnya diperlukan untuk tugas khusus | Umumnya diperlukan untuk tugas khusus |

> **Catatan:** Kinerja tidak dijamin hanya karena model berbahasa Indonesia. Hasil tetap bergantung pada kualitas data, kecocokan domain, label, pembagian dataset, konfigurasi pelatihan, dan evaluasi.

## 4. Cara Kerja IndoBERT

Gambaran alur klasifikasi teks:

```text
Teks
  ↓
Tokenizer IndoBERT
  ↓
Input IDs + Attention Mask
  ↓
Transformer Encoder
  ↓
Representasi Kontekstual
  ↓
Classification Head
  ↓
Probabilitas Kelas
  ↓
Label Prediksi
```

### 4.1 Pre-training

Pada tahap pre-training, model mempelajari pola bahasa dari korpus teks dalam skala besar menggunakan tujuan pelatihan yang dirancang untuk mempelajari representasi bahasa.

### 4.2 Fine-tuning

Pada tahap fine-tuning, model pretrained dilatih lebih lanjut dengan dataset berlabel untuk tugas tertentu, misalnya:

- `0`: Non-Emergency
- `1`: Emergency

Model menyesuaikan parameter agar representasinya berguna untuk membedakan kelas yang ditentukan.

### 4.3 Classification head

Classification head adalah lapisan tambahan yang mengubah representasi model menjadi skor kelas. Untuk klasifikasi dua kelas, model menghasilkan dua logit yang dapat diubah menjadi probabilitas.

Contoh ilustratif:

```text
Logits:
[-0.8, 2.1]

Probabilitas setelah softmax:
[0.052, 0.948]

Prediksi:
Emergency
```

Angka di atas hanya contoh ilustratif, bukan hasil pengukuran dari model tertentu.

## 5. Tokenizer IndoBERT

Model tidak menerima kalimat mentah secara langsung. Tokenizer mengubah teks menjadi unit token yang dikenal model, kemudian memetakannya menjadi ID numerik.

Contoh konseptual:

```text
Teks:
"Tolong bantu kami"

Token:
[CLS] tolong bantu kami [SEP]

Input IDs:
[101, ..., ..., ..., 102]
```

ID token pada contoh di atas hanya ilustrasi. ID sebenarnya harus diperoleh dari tokenizer checkpoint yang digunakan.

### 5.1 Token khusus

Token khusus yang umum ditemukan pada BERT antara lain:

| Token | Fungsi umum |
| --- | --- |
| `[CLS]` | Token khusus di awal input; representasinya sering digunakan untuk klasifikasi |
| `[SEP]` | Menandai pemisah atau akhir segmen teks |
| `[PAD]` | Mengisi posisi kosong agar panjang input seragam dalam satu batch |
| `[UNK]` | Mewakili token yang tidak dapat dipetakan ke token yang dikenal, jika digunakan oleh tokenizer |
| `[MASK]` | Digunakan dalam skema masked language modeling |

Token dan perilaku tokenizer mengikuti checkpoint yang dipakai. Periksa hasil aktual menggunakan `tokenizer`.

### 5.2 Input IDs

`input_ids` adalah daftar ID numerik yang mewakili token.

### 5.3 Attention mask

`attention_mask` umumnya menggunakan:

- `1` untuk token yang perlu diperhatikan model.
- `0` untuk token padding yang harus diabaikan.

### 5.4 Padding dan truncation

- **Padding:** menambahkan token `[PAD]` agar input dalam batch memiliki panjang yang seragam.
- **Truncation:** memotong input yang melebihi batas panjang yang ditentukan.

Batas panjang yang sesuai bergantung pada checkpoint dan kebutuhan tugas. Input yang terlalu panjang dapat kehilangan informasi saat dipotong.

## 6. Menyiapkan Environment

Disarankan menggunakan virtual environment atau environment Conda tersendiri.

Instalasi paket:

```bash
pip install torch transformers scikit-learn pandas
```

Untuk eksperimen yang lebih besar, periksa kompatibilitas versi PyTorch, CUDA, dan perangkat keras yang digunakan. Contoh pada materi ini dapat dijalankan pada CPU, tetapi fine-tuning akan lebih lambat dibandingkan GPU.

Periksa versi dan perangkat:

```python
import torch
import transformers

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA tersedia:", torch.cuda.is_available())
```

Contoh output (versi akan berbeda sesuai environment):

```text
PyTorch: 2.x.x
Transformers: 4.x.x
CUDA tersedia: False
```

Output di atas hanya contoh format; angka versi aktual bergantung pada instalasi Anda.

## 7. Memuat Tokenizer dan Model

Contoh checkpoint IndoBERT:

`indobenchmark/indobert-base-p1`

Checkpoint harus tersedia di Hugging Face Hub atau sudah tersimpan secara lokal. Pada pemakaian pertama, library biasanya mengunduh file model sehingga koneksi internet diperlukan.

```python
from transformers import AutoTokenizer, AutoModel

model_name = "indobenchmark/indobert-base-p1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print("Tokenizer dan model berhasil dimuat.")
print("Ukuran vocabulary:", tokenizer.vocab_size)
print("Hidden size:", model.config.hidden_size)
```

Contoh output:

```text
Tokenizer dan model berhasil dimuat.
Ukuran vocabulary: <nilai dari checkpoint>
Hidden size: <nilai dari konfigurasi checkpoint>
```

Nilai dalam tanda kurung sudut sengaja tidak ditetapkan karena output harus mengikuti konfigurasi checkpoint yang benar-benar berhasil dimuat. Jalankan kode untuk mendapatkan nilai aktual.

### Apa perbedaan `AutoModel` dan `AutoModelForSequenceClassification`?

- `AutoModel`: memuat backbone Transformer dan menghasilkan representasi tersembunyi. Berguna untuk mempelajari output model atau membangun komponen sendiri.
- `AutoModelForSequenceClassification`: memuat backbone beserta classification head untuk klasifikasi teks.

Untuk proyek klasifikasi teks, `AutoModelForSequenceClassification` biasanya lebih praktis.

## 8. Memahami Hasil Tokenisasi

```python
from transformers import AutoTokenizer

model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "Tolong bantu kami, rumah terendam banjir."

encoded = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=16,
    return_tensors="pt"
)

print("Input IDs:")
print(encoded["input_ids"])

print("\nAttention mask:")
print(encoded["attention_mask"])

print("\nToken:")
print(tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))
```

Output aktual bergantung pada tokenizer dan versi paket. Secara umum:

```text
Input IDs:
tensor([[ID_CLS, ID_TOLONG, ID_BANTU, ..., ID_SEP, ID_PAD, ...]])

Attention mask:
tensor([[1, 1, 1, ..., 1, 0, ...]])

Token:
['[CLS]', '...', '...', ..., '[SEP]', '[PAD]', ...]
```

`ID_CLS` dan simbol lain di atas adalah placeholder penjelasan, bukan nilai literal yang dapat digunakan dalam kode. Program akan mencetak ID dan token yang sebenarnya.

### Periksa bentuk tensor

```python
print("Shape input_ids:", encoded["input_ids"].shape)
print("Shape attention_mask:", encoded["attention_mask"].shape)
```

Output:

```text
Shape input_ids: torch.Size([1, 16])
Shape attention_mask: torch.Size([1, 16])
```

Penjelasan:

- `1` adalah jumlah teks dalam batch.
- `16` adalah panjang urutan yang ditetapkan melalui `max_length`.

## 9. Menghasilkan Representasi Teks

Dengan `AutoModel`, kita dapat mengambil representasi tersembunyi dari token.

```python
import torch
from transformers import AutoTokenizer, AutoModel

model_name = "indobenchmark/indobert-base-p1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

text = "Tolong bantu kami, rumah terendam banjir."
inputs = tokenizer(text, return_tensors="pt", truncation=True)

with torch.no_grad():
    outputs = model(**inputs)

print("Bentuk last_hidden_state:", outputs.last_hidden_state.shape)
```

Untuk model BERT-base dengan hidden size 768, output biasanya berbentuk:

```text
Bentuk last_hidden_state: torch.Size([1, jumlah_token, 768])
```

`jumlah_token` bergantung pada tokenisasi teks. Nilai `768` berlaku untuk konfigurasi base yang umum, tetapi selalu periksa `model.config.hidden_size` untuk memastikan.

### Apa arti `last_hidden_state`?

Bentuk tensor umumnya:

```text
(batch_size, sequence_length, hidden_size)
```

Setiap posisi token memiliki vektor representasi kontekstual. Vektor ini bukan probabilitas kelas dan tidak otomatis menunjukkan tingkat kepentingan kata.

Untuk klasifikasi, gunakan model dengan classification head atau latih classifier pada representasi yang sesuai.

## 10. Fine-Tuning untuk Klasifikasi Teks

Pada bagian ini, kita melatih IndoBERT untuk membedakan teks `Emergency` dan `Non-Emergency`.

### 10.1 Siapkan dataset contoh

Dataset berikut **hanya data mainan untuk demonstrasi kode**, bukan dataset penelitian yang memadai.

```python
texts = [
    "Tolong bantu kami, rumah terendam banjir",
    "Kami terjebak banjir dan membutuhkan evakuasi",
    "Mohon bantuan, ada warga terluka",
    "Hujan turun sejak pagi di kota ini",
    "Saya sedang membaca berita banjir",
    "Cuaca hari ini cukup cerah",
    "Kami membutuhkan makanan dan air bersih",
    "Pemerintah mengumumkan prakiraan cuaca"
]

# 1 = Emergency, 0 = Non-Emergency
labels = [1, 1, 1, 0, 0, 0, 1, 0]
```

Output:

```text
Jumlah data: 8
Jumlah kelas Emergency: 4
Jumlah kelas Non-Emergency: 4
```

### 10.2 Bagi data

Untuk dataset nyata, pembagian train/validation/test perlu direncanakan dengan hati-hati. Data yang hampir duplikat atau berasal dari peristiwa yang sama dapat menyebabkan data leakage jika tersebar ke beberapa bagian.

Contoh berikut hanya untuk memperlihatkan cara kerja API pembagian data:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)

print("Data train:", len(X_train))
print("Data test:", len(X_test))
```

Output:

```text
Data train: 6
Data test: 2
```

Delapan contoh terlalu sedikit untuk menghasilkan kesimpulan ilmiah. Gunakan dataset yang cukup besar dan representatif untuk eksperimen sebenarnya.

### 10.3 Buat dataset PyTorch

```python
import torch
from torch.utils.data import Dataset

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoded.items()
        }
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
```

Kelas ini mengubah setiap teks menjadi tensor yang dapat diproses oleh model.

### 10.4 Muat model klasifikasi

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "indobenchmark/indobert-base-p1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
```

Saat classification head baru dibuat, bobot head biasanya diinisialisasi baru. Pesan peringatan bahwa sebagian bobot classification head baru diinisialisasi dapat muncul dan bukan otomatis berarti proses gagal. Fine-tuning diperlukan agar head belajar dari label dataset.

### 10.5 Fine-tuning menggunakan Trainer

Contoh berikut menunjukkan kerangka training Hugging Face. Untuk dataset sungguhan, tambahkan validation set, evaluasi per epoch, penyimpanan checkpoint, dan pengaturan eksperimen.

```python
import numpy as np
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

texts = [
    "Tolong bantu kami, rumah terendam banjir",
    "Kami terjebak banjir dan membutuhkan evakuasi",
    "Mohon bantuan, ada warga terluka",
    "Kami membutuhkan makanan dan air bersih",
    "Hujan turun sejak pagi di kota ini",
    "Saya sedang membaca berita banjir",
    "Cuaca hari ini cukup cerah",
    "Pemerintah mengumumkan prakiraan cuaca",
    "Ada warga yang membutuhkan pertolongan segera",
    "Informasi cuaca untuk besok telah diterbitkan",
    "Jalan terputus dan kami belum bisa keluar",
    "Berita mengenai kegiatan sekolah hari ini"
]
labels = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def make_hf_dataset(text_list, label_list):
    dataset = HFDataset.from_dict({
        "text": text_list,
        "labels": label_list
    })

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    return dataset.map(tokenize_batch, batched=True).remove_columns(["text"])

train_dataset = make_hf_dataset(X_train, y_train)
test_dataset = make_hf_dataset(X_test, y_test)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "Non-Emergency", 1: "Emergency"},
    label2id={"Non-Emergency": 0, "Emergency": 1}
)

def compute_metrics(eval_pred):
    logits, labels_true = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_true, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels_true, predictions),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1
    }

training_args = TrainingArguments(
    output_dir="./indobert-emergency-model",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
print(trainer.evaluate())
```

**Catatan kompatibilitas:** nama argumen pada Hugging Face Transformers dapat berbeda menurut versi. Pada versi yang lebih lama, `eval_strategy` mungkin perlu diganti menjadi `evaluation_strategy`. Periksa versi yang terpasang jika muncul `unexpected keyword argument`.

Output training akan bervariasi, misalnya:

```text
{'loss': ..., 'grad_norm': ..., 'learning_rate': ..., 'epoch': ...}
{'eval_loss': ..., 'eval_accuracy': ..., 'eval_precision_macro': ...,
 'eval_recall_macro': ..., 'eval_f1_macro': ...}
```

Titik-titik adalah placeholder karena nilai sebenarnya hanya tersedia setelah kode dijalankan. **Jangan menggunakan skor dari dataset mainan sebagai hasil penelitian.** Pada contoh ini jumlah data sangat kecil, sehingga metrik tidak stabil dan tidak dapat digeneralisasi.

Untuk eksperimen yang lebih baik, sisihkan train, validation, dan test secara terpisah. Gunakan validation untuk memilih konfigurasi dan test hanya untuk evaluasi akhir.

## 11. Evaluasi Model

Metrik yang perlu dipahami:

| Metrik | Makna |
| --- | --- |
| Accuracy | Proporsi seluruh prediksi yang benar |
| Precision | Dari semua prediksi suatu kelas, berapa proporsi yang benar |
| Recall | Dari seluruh data aktual suatu kelas, berapa proporsi yang berhasil ditemukan |
| F1-score | Rata-rata harmonik precision dan recall |
| Macro F1 | Rata-rata F1 setiap kelas dengan bobot sama |
| Weighted F1 | Rata-rata F1 dengan bobot sesuai jumlah data per kelas |

Untuk deteksi permintaan bantuan darurat, **Recall kelas Emergency** penting karena false negative berarti permintaan bantuan yang sebenarnya tidak terdeteksi. Namun, precision juga perlu diperiksa agar sistem tidak membanjiri petugas dengan terlalu banyak peringatan yang keliru.

### Menghitung metrik dari prediksi

```python
from sklearn.metrics import classification_report, confusion_matrix

y_true = [1, 1, 0, 0, 1, 0]
y_pred = [1, 0, 0, 0, 1, 1]

print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["Non-Emergency", "Emergency"],
    zero_division=0
))
```

Output:

```text
[[2 1]
 [1 2]]
```

Confusion matrix memakai urutan label `[0, 1]`, sehingga baris adalah label aktual dan kolom adalah prediksi:

| | Prediksi Non-Emergency | Prediksi Emergency |
| --- | ---: | ---: |
| Aktual Non-Emergency | 2 (TN) | 1 (FP) |
| Aktual Emergency | 1 (FN) | 2 (TP) |

Classification report akan menampilkan precision, recall, F1-score, dan support untuk tiap kelas. Nilai dihitung oleh kode berdasarkan contoh di atas.

## 12. Membuat Prediksi Teks Baru

Setelah fine-tuning, simpan model dan tokenizer:

```python
trainer.save_model("./indobert-emergency-model")
tokenizer.save_pretrained("./indobert-emergency-model")
```

Memuat kembali model:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_dir = "./indobert-emergency-model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

text = "Tolong bantu kami, rumah sudah terendam banjir."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)
    predicted_id = int(torch.argmax(probabilities, dim=-1).item())

print("Probabilitas:", probabilities[0].tolist())
print("Prediksi:", model.config.id2label[predicted_id])
```

Output ilustratif:

```text
Probabilitas: [0.08, 0.92]
Prediksi: Emergency
```

**Angka di atas hanya ilustrasi.** Prediksi aktual bergantung pada model hasil fine-tuning, data, dan konfigurasi. Jangan mengasumsikan contoh teks pasti akan memperoleh hasil tersebut.

Probabilitas softmax bukan jaminan bahwa confidence model terkalibrasi dengan baik. Untuk penggunaan operasional, lakukan validasi dan pertimbangkan kalibrasi probabilitas serta ambang keputusan.

## 13. Preprocessing untuk IndoBERT

Satu kesalahan umum adalah menerapkan semua preprocessing klasik secara otomatis pada input BERT.

### TF-IDF dibandingkan IndoBERT

| Langkah | TF-IDF + SVM | IndoBERT |
| --- | --- | --- |
| Menghapus URL/noise yang tidak relevan | Umumnya berguna | Bisa berguna, bergantung pada data |
| Lowercase | Umumnya lazim jika konfigurasi mendukung | Ikuti tokenizer; jangan mengubah teks secara sembarang |
| Stopword removal | Dapat diuji sebagai bagian eksperimen | Umumnya tidak perlu secara default |
| Stemming | Dapat diuji sebagai bagian eksperimen | Umumnya tidak perlu secara default |
| Tokenisasi | Tokenisasi kata/fitur | Gunakan tokenizer checkpoint |
| Menghapus kata negasi | Berisiko mengubah makna | Hindari; dapat mengubah makna penting |
| Padding dan truncation | Tidak seperti input BERT | Diperlukan sesuai strategi batching/model |

Contoh, kalimat:

`"Saya tidak membutuhkan bantuan"`

Jika kata `tidak` dihapus, makna kalimat bisa berubah. Karena itu, jangan menghapus stopword secara membabi buta.

**Prinsip:** mulai dari teks yang relatif bersih, pertahankan informasi bermakna, lalu gunakan tokenizer bawaan IndoBERT. Uji keputusan preprocessing sebagai eksperimen, bukan asumsi.

## 14. IndoBERT dalam Penelitian Bantuan Darurat

Target penelitian yang relevan:

**Deteksi Tweet Permintaan Bantuan Darurat saat Bencana Menggunakan IndoBERT.**

### 14.1 Definisi kelas awal

| Label | ID | Definisi operasional awal |
| --- | ---: | --- |
| Non-Emergency | 0 | Teks informatif atau percakapan yang tidak menunjukkan permintaan bantuan darurat berdasarkan pedoman anotasi |
| Emergency | 1 | Teks yang menunjukkan kebutuhan bantuan/pertolongan mendesak atau kondisi berbahaya sesuai pedoman anotasi |

Definisi tersebut harus diperjelas sebelum anotasi. Misalnya, tweet “Banjir terjadi di desa X” belum tentu merupakan permintaan bantuan langsung; keputusan label harus konsisten dengan tujuan penelitian dan pedoman anotasi.

### 14.2 Pipeline penelitian

```text
Pengumpulan Data yang Sah
          ↓
Penyaringan dan Penghapusan Duplikat
          ↓
Anotasi dengan Pedoman yang Jelas
          ↓
Pemeriksaan Kualitas Label
          ↓
Train / Validation / Test Split
          ↓
Baseline: TF-IDF + Logistic Regression
          ↓
Baseline: TF-IDF + SVM
          ↓
Fine-Tuning IndoBERT
          ↓
Evaluasi dan Analisis Kesalahan
          ↓
Perbandingan Model
          ↓
Model Terpilih
          ↓
FastAPI / REST API
```

Pastikan pengumpulan data mematuhi ketentuan platform, kebijakan privasi, dan persetujuan yang diperlukan. Hindari menyebarkan data pribadi atau lokasi sensitif yang tidak perlu.

### 14.3 Baseline wajib

Bandingkan IndoBERT dengan model yang lebih sederhana, misalnya:

1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM
3. IndoBERT fine-tuned

Gunakan pembagian data dan prosedur evaluasi yang sebanding. Laporkan precision, recall, F1 per kelas, macro F1, confusion matrix, dan bila relevan biaya komputasi.

### 14.4 Hindari data leakage

Jangan melakukan hal-hal berikut:

- Menempatkan duplikat tweet pada train dan test sekaligus.
- Menggunakan test set untuk memilih hyperparameter.
- Membuat keputusan preprocessing berdasarkan hasil test berulang kali.
- Mengabaikan tweet yang berasal dari kejadian yang sama jika hal itu membuat evaluasi terlalu optimistis.

Untuk data bencana, pertimbangkan pembagian berdasarkan waktu atau kejadian bencana bila sesuai dengan pertanyaan penelitian. Ini dapat menguji kemampuan model menghadapi peristiwa yang belum pernah dilihat.

### 14.5 Catatan tentang lokasi dan prioritas

IndoBERT mengklasifikasikan teks; model tidak otomatis memverifikasi lokasi, tingkat keparahan, atau kebenaran laporan. Informasi lokasi dan prioritas perlu diproses dan divalidasi secara terpisah. Prediksi model sebaiknya menjadi alat bantu triase, bukan pengganti verifikasi petugas.

## 15. Kesalahan yang Sering Terjadi

1. **Mengira IndoBERT siap untuk tugas khusus tanpa fine-tuning.** Model dasar belum tentu mempunyai label tugas yang sesuai.
2. **Menggunakan tokenizer yang tidak cocok dengan model.** Selalu gunakan tokenizer dari checkpoint yang sama.
3. **Menghapus stopword tanpa pertimbangan.** Negasi seperti “tidak” dapat mengubah makna.
4. **Hanya melaporkan accuracy.** Pada kelas tidak seimbang, accuracy dapat menyesatkan.
5. **Menganggap softmax sebagai confidence yang pasti benar.** Probabilitas perlu divalidasi.
6. **Melakukan evaluasi pada dataset terlalu kecil.** Skor tidak stabil dan tidak mewakili kinerja di dunia nyata.
7. **Mengabaikan data leakage.** Duplikat atau tweet dari peristiwa yang sama dapat membuat hasil tampak lebih baik.
8. **Menganggap model mengetahui lokasi atau urgensi faktual.** Klasifikasi teks tidak memverifikasi fakta di lapangan.
9. **Mengabaikan keterbatasan komputasi.** Fine-tuning model Transformer dapat membutuhkan memori dan waktu yang cukup besar.

## 16. Latihan

Kerjakan latihan berikut secara berurutan:

1. Muat tokenizer IndoBERT dan tampilkan token dari tiga kalimat bahasa Indonesia.
2. Bandingkan hasil tokenisasi kata biasa dengan tokenisasi IndoBERT.
3. Cetak `input_ids`, `attention_mask`, dan bentuk tensor untuk teks dengan panjang berbeda.
4. Jelaskan fungsi `[CLS]`, `[SEP]`, `[PAD]`, dan `[MASK]`.
5. Jalankan contoh `AutoModel` dan jelaskan bentuk `last_hidden_state`.
6. Fine-tune `AutoModelForSequenceClassification` pada dataset latihan.
7. Hitung confusion matrix dan recall kelas Emergency.
8. Bandingkan hasil IndoBERT dengan TF-IDF + Logistic Regression atau SVM.
9. Uji apakah pembersihan URL atau normalisasi slang membantu hasil validasi.
10. Dokumentasikan contoh false positive dan false negative serta kemungkinan penyebabnya.

## 17. Mini Project

### Judul

**Klasifikasi Tweet Permintaan Bantuan Darurat Menggunakan IndoBERT**

### Tujuan

Membangun prototipe klasifikasi teks yang membedakan laporan permintaan bantuan darurat dari teks non-darurat.

### Tahapan

1. Siapkan dataset yang sumber dan penggunaannya dapat dipertanggungjawabkan.
2. Buat pedoman anotasi Emergency dan Non-Emergency.
3. Lakukan anotasi dan pemeriksaan kualitas label.
4. Hapus duplikat dan lakukan analisis distribusi kelas.
5. Bagi data menjadi train, validation, dan test.
6. Bangun baseline TF-IDF + Logistic Regression dan TF-IDF + SVM.
7. Fine-tune IndoBERT.
8. Evaluasi dengan precision, recall, F1, macro F1, dan confusion matrix.
9. Analisis false positive dan false negative.
10. Simpan model dan tokenizer untuk dipakai oleh aplikasi.
11. Jika model sudah dievaluasi, buat endpoint FastAPI untuk inferensi.
12. Dokumentasikan keterbatasan, etika data, dan risiko penggunaan.

### Kriteria keberhasilan

- Dataset dan label terdokumentasi.
- Proses pembagian data dapat direproduksi.
- Baseline dan IndoBERT dievaluasi dengan prosedur yang adil.
- Recall Emergency dan F1 dilaporkan, bukan accuracy saja.
- Model tersimpan bersama tokenizer dan pemetaan label.
- Sistem tidak mengklaim bahwa prediksi otomatis membuktikan keadaan darurat.

## 18. Checkpoint

Sebelum melanjutkan ke deployment, pastikan Anda dapat mencentang semua poin berikut:

- [ ] Saya dapat menjelaskan apa itu IndoBERT dan hubungannya dengan BERT.
- [ ] Saya memahami tokenizer dan token khusus.
- [ ] Saya dapat menjelaskan `input_ids` dan `attention_mask`.
- [ ] Saya dapat memuat model dari Hugging Face.
- [ ] Saya memahami perbedaan `AutoModel` dan `AutoModelForSequenceClassification`.
- [ ] Saya dapat melakukan fine-tuning pada dataset berlabel.
- [ ] Saya dapat menghitung precision, recall, F1, dan confusion matrix.
- [ ] Saya memahami mengapa recall kelas Emergency penting.
- [ ] Saya dapat membandingkan IndoBERT dengan baseline TF-IDF.
- [ ] Saya memahami risiko data leakage dan keterbatasan data kecil.
- [ ] Saya dapat menyimpan serta memuat kembali model dan tokenizer.

## 19. Referensi

- Hugging Face Transformers documentation: <https://huggingface.co/docs/transformers/>
- Hugging Face model page — IndoBERT: <https://huggingface.co/indobenchmark/indobert-base-p1>
- PyTorch documentation: <https://pytorch.org/docs/stable/index.html>
- Scikit-learn model evaluation: <https://scikit-learn.org/stable/modules/model_evaluation.html>
- Koto, F., Rahimi, A., Lau, J. H., & Baldwin, T. (2020). *IndoLEM and IndoBERT: A Benchmark Dataset and Pre-trained Language Model for Indonesian NLP*. Proceedings of the 28th International Conference on Computational Linguistics.

---

## Ringkasan

IndoBERT merupakan model berbasis BERT yang dirancang untuk memahami teks bahasa Indonesia. Untuk menggunakannya pada klasifikasi, alur umumnya adalah:

```text
Dataset Berlabel
      ↓
Tokenizer IndoBERT
      ↓
Fine-Tuning Model
      ↓
Evaluasi yang Benar
      ↓
Analisis Kesalahan
      ↓
Simpan Model + Tokenizer
      ↓
Integrasi ke REST API
```

Kunci keberhasilan bukan hanya memilih IndoBERT, tetapi juga membuat label yang jelas, menghindari data leakage, memilih metrik yang sesuai, membandingkan baseline, dan memvalidasi model pada data yang mewakili penggunaan sebenarnya.
