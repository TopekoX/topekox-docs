# Fine-Tuning IndoBERT untuk Klasifikasi Teks Bahasa Indonesia

> Materi pembelajaran NLP — dari konsep fine-tuning hingga evaluasi model klasifikasi teks menggunakan Hugging Face Transformers.

## Tujuan Pembelajaran

Setelah mempelajari materi ini, Anda diharapkan mampu:

1. Menjelaskan perbedaan pre-training dan fine-tuning.
2. Memahami komponen IndoBERT untuk klasifikasi teks.
3. Menyiapkan dataset berlabel.
4. Melakukan tokenisasi menggunakan tokenizer IndoBERT.
5. Melatih model klasifikasi dengan Hugging Face Transformers.
6. Mengevaluasi hasil menggunakan precision, recall, F1-score, dan confusion matrix.
7. Menyimpan model dan menggunakannya untuk memprediksi teks baru.

Contoh penelitian yang digunakan adalah **klasifikasi tweet permintaan bantuan darurat saat bencana** dengan dua label:

- `0`: `Non-Emergency`
- `1`: `Emergency`

Dataset dalam materi ini hanyalah data contoh untuk pembelajaran, bukan dataset penelitian yang telah divalidasi.

---

## 1. Apa Itu Fine-Tuning?

Fine-tuning adalah proses melatih lebih lanjut model yang sudah melalui pre-training menggunakan dataset untuk tugas tertentu.

Contoh alur:

```text
Kumpulan teks bahasa Indonesia
          |
          v
  Pre-training IndoBERT
          |
          v
  Model IndoBERT pralatih
          |
          v
  Fine-tuning dengan data berlabel
          |
          v
  Model klasifikasi Emergency
```

### Pre-training vs Fine-tuning

| Aspek | Pre-training | Fine-tuning |
|---|---|---|
| Tujuan | Mempelajari pola bahasa umum | Menyesuaikan model untuk tugas tertentu |
| Data | Umumnya teks besar tanpa label tugas | Dataset yang memiliki label |
| Biaya komputasi | Biasanya sangat besar | Lebih kecil, tetapi bergantung pada data dan perangkat |
| Contoh | Mempelajari pola bahasa Indonesia | Mengklasifikasikan tweet Emergency/Non-Emergency |

Fine-tuning tidak berarti membuat model dari nol. Kita memanfaatkan bobot model pralatih lalu menyesuaikannya dengan data tugas.

## 2. Mengapa Menggunakan IndoBERT?

IndoBERT adalah keluarga model BERT yang dilatih untuk memahami bahasa Indonesia. BERT menggunakan representasi kontekstual: makna token dipengaruhi oleh konteks kalimatnya.

Misalnya, kata “bantuan” pada kalimat berikut perlu dipahami bersama konteksnya:

- “Kami membutuhkan bantuan, rumah terendam banjir.”
- “Lembaga itu mengumumkan program bantuan pendidikan.”

Untuk klasifikasi, model IndoBERT ditambah *classification head* yang menghasilkan skor untuk setiap kelas.

```text
Teks
 |
 v
Tokenizer IndoBERT
 |
 v
Input IDs + Attention Mask
 |
 v
IndoBERT Encoder
 |
 v
Classification Head
 |
 v
Skor kelas
 |
 v
Emergency / Non-Emergency
```

> Catatan: model yang dipakai pada contoh ini adalah `indobenchmark/indobert-base-p1`. Sebelum eksperimen penelitian, periksa model card dan lisensi model tersebut, serta pastikan kecocokannya dengan tugas dan lingkungan Anda.

## 3. Konsep Penting Sebelum Praktik

### 3.1 Tokenizer

Tokenizer memecah teks menjadi token/subword dan mengubahnya menjadi ID numerik yang dikenal model.

### 3.2 Input IDs

Deretan ID token yang menjadi masukan model.

### 3.3 Attention Mask

Menandai token yang benar-benar merupakan masukan (`1`) dan padding (`0`).

### 3.4 Padding dan Truncation

- **Padding** menambahkan token pengisi agar panjang input seragam dalam satu batch.
- **Truncation** memotong teks yang melebihi batas panjang yang ditentukan.

### 3.5 Label

Label adalah target yang dipelajari model. Pastikan pemetaan label konsisten di seluruh proses.

### 3.6 Train, Validation, dan Test

- **Train**: digunakan untuk memperbarui bobot model.
- **Validation**: membantu memilih konfigurasi/model selama eksperimen.
- **Test**: digunakan untuk evaluasi akhir, idealnya hanya setelah keputusan model ditetapkan.

Jangan menggunakan data test untuk memilih hyperparameter karena dapat menyebabkan kebocoran informasi (*data leakage*).

---

## 4. Menyiapkan Environment

Disarankan menggunakan Python environment terpisah atau Google Colab. Fine-tuning dapat berjalan di CPU, tetapi biasanya lebih lambat daripada GPU.

Instal dependensi:

```bash
pip install -U torch transformers datasets evaluate scikit-learn pandas accelerate
```

Jika memakai Jupyter Notebook, jalankan perintah instalasi di sel notebook dengan awalan `!`:

```python
!pip install -U torch transformers datasets evaluate scikit-learn pandas accelerate
```

Cek versi dan ketersediaan GPU:

```python
import torch
import transformers

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA tersedia:", torch.cuda.is_available())
```

Contoh output (versi dan status GPU bergantung pada lingkungan):

```text
PyTorch: 2.x.x
Transformers: 4.x.x
CUDA tersedia: True
```

Jika `CUDA tersedia: False`, kode tetap dapat dijalankan pada CPU, tetapi proses training mungkin lebih lambat.

---

## 5. Membuat Dataset Contoh

Untuk pembelajaran, kita gunakan data sintetis kecil. Data ini **tidak cukup untuk menyimpulkan performa penelitian**. Hasil model pada dataset kecil dapat tidak stabil dan tidak mewakili kondisi nyata.

```python
import pandas as pd

data = [
    ("Tolong bantu kami, rumah terendam banjir.", 1),
    ("Kami terjebak banjir dan membutuhkan evakuasi.", 1),
    ("Mohon kirim bantuan medis ke desa kami.", 1),
    ("Jalan menuju kampung terputus akibat longsor.", 1),
    ("Ada warga terluka setelah gempa, perlu pertolongan.", 1),
    ("Air terus naik, keluarga kami belum bisa keluar.", 1),
    ("Kami kehabisan makanan setelah banjir.", 1),
    ("Mohon bantuan untuk warga yang terisolasi.", 1),
    ("Hujan turun sejak pagi di kota ini.", 0),
    ("Berita tentang banjir disiarkan sore ini.", 0),
    ("Pemerintah mengadakan simulasi evakuasi minggu depan.", 0),
    ("Saya membaca artikel tentang mitigasi bencana.", 0),
    ("Cuaca hari ini mendung dan berangin.", 0),
    ("Relawan mengadakan pelatihan kebencanaan.", 0),
    ("Foto banjir tahun lalu kembali dibagikan.", 0),
    ("Semoga semua warga tetap waspada saat musim hujan.", 0),
]

df = pd.DataFrame(data, columns=["text", "label"])
df["label_name"] = df["label"].map({0: "Non-Emergency", 1: "Emergency"})

print(df.head())
print("\\nDistribusi label:")
print(df["label_name"].value_counts())
```

Contoh output:

```text
                                                text  label     label_name
0       Tolong bantu kami, rumah terendam banjir.      1      Emergency
1  Kami terjebak banjir dan membutuhkan evakuasi.      1      Emergency
2          Mohon kirim bantuan medis ke desa kami.      1      Emergency
3   Jalan menuju kampung terputus akibat longsor.      1      Emergency
4  Ada warga terluka setelah gempa, perlu pertolongan. 1      Emergency

Distribusi label:
Emergency        8
Non-Emergency    8
Name: count, dtype: int64
```

> Output tabel dapat berbeda sedikit berdasarkan versi pandas dan format tampilan notebook.

### Hal yang perlu diperhatikan pada dataset nyata

- Buat pedoman anotasi yang eksplisit untuk membedakan permintaan bantuan, informasi kejadian, dan berita umum.
- Hindari duplikasi teks antar-split.
- Jika beberapa tweet berasal dari peristiwa, akun, atau waktu yang sama, pertimbangkan split berbasis kelompok/waktu agar evaluasi tidak terlalu optimistis.
- Lindungi informasi pribadi dan patuhi ketentuan platform serta persetujuan yang berlaku.
- Simpan sumber, waktu pengambilan, dan proses anotasi secara terdokumentasi.

## 6. Membagi Dataset

Untuk dataset kecil ini, split hanya mendemonstrasikan proses teknis. Dalam penelitian, gunakan dataset yang memadai dan pertimbangkan split berbasis waktu atau kelompok jika diperlukan.

```python
from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(
    df[["text", "label"]],
    test_size=0.4,
    random_state=42,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["label"]
)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))
```

Contoh output:

```text
Train: 9
Validation: 3
Test: 4
```

Karena data hanya 16 baris, metrik test akan sangat tidak stabil. Angka tersebut bukan ukuran dataset yang disarankan untuk penelitian.

## 7. Memuat Tokenizer IndoBERT

```python
from transformers import AutoTokenizer

MODEL_NAME = "indobenchmark/indobert-base-p1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = "Tolong bantu kami, rumah terendam banjir."
encoded = tokenizer(
    text,
    truncation=True,
    padding="max_length",
    max_length=32,
    return_tensors="pt"
)

print("Input IDs:")
print(encoded["input_ids"])
print("Attention Mask:")
print(encoded["attention_mask"])
print("Token hasil tokenisasi:")
print(tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))
```

Contoh bentuk output (ID token dapat bergantung pada tokenizer/model):

```text
Input IDs:
tensor([[...]])

Attention Mask:
tensor([[1, 1, 1, ..., 0, 0]])

Token hasil tokenisasi:
['[CLS]', 'tolong', 'bantu', 'kami', ',', 'rumah', 'terendam', 'banjir', '.', '[SEP]', '[PAD]', ...]
```

`...` sengaja digunakan untuk meringkas deretan ID yang panjang. Gunakan output runtime untuk melihat ID token yang persis. Tokenisasi subword tidak selalu sama dengan pemisahan kata berdasarkan spasi.

### Mengapa `max_length=32`?

Angka ini hanya contoh untuk teks pendek. Untuk dataset nyata, periksa distribusi panjang token dan pilih batas yang sesuai. Batas terlalu panjang meningkatkan penggunaan memori; batas terlalu pendek dapat membuang informasi penting.

---

## 8. Mengubah Data ke Format Dataset Hugging Face

Kita gunakan `Dataset` agar data mudah diproses oleh `Trainer`.

```python
from datasets import Dataset, DatasetDict

raw_datasets = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
    "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
})

print(raw_datasets)
```

Contoh output:

```text
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 9
    })
    validation: Dataset({
        features: ['text', 'label'],
        num_rows: 3
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 4
    })
})
```

## 9. Tokenisasi Dataset

```python
def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128
    )

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True
)

print(tokenized_datasets["train"].column_names)
```

Contoh output:

```text
['text', 'label', 'input_ids', 'token_type_ids', 'attention_mask']
```

Kolom dapat sedikit berbeda tergantung tokenizer. Padding akan ditangani secara dinamis oleh data collator, sehingga setiap batch dipad hingga panjang maksimum dalam batch tersebut.

## 10. Memuat Model untuk Klasifikasi

Gunakan `AutoModelForSequenceClassification`, bukan hanya `AutoModel`, karena tugas kita adalah klasifikasi urutan teks.

```python
from transformers import AutoModelForSequenceClassification

id2label = {
    0: "Non-Emergency",
    1: "Emergency"
}
label2id = {
    "Non-Emergency": 0,
    "Emergency": 1
}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)
```

Saat pertama kali memuat model untuk klasifikasi, pustaka dapat memberi peringatan bahwa bobot *classification head* baru diinisialisasi. Itu normal: head tersebut perlu dipelajari melalui fine-tuning. Bobot encoder berasal dari model pralatih.

## 11. Menyiapkan Metrik Evaluasi

Untuk klasifikasi permintaan bantuan, jangan hanya melihat accuracy. Recall kelas `Emergency` penting karena false negative berarti tweet yang membutuhkan bantuan tidak terdeteksi.

```python
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        pos_label=1,
        zero_division=0
    )

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_emergency": precision,
        "recall_emergency": recall,
        "f1_emergency": f1
    }
```

Makna metrik:

- **Precision Emergency**: dari semua teks yang diprediksi Emergency, berapa yang benar-benar Emergency.
- **Recall Emergency**: dari semua teks yang sebenarnya Emergency, berapa yang berhasil ditemukan model.
- **F1 Emergency**: rata-rata harmonik precision dan recall.
- **Accuracy**: proporsi semua prediksi yang benar; dapat menyesatkan pada data tidak seimbang.

## 12. Konfigurasi Fine-Tuning

Parameter berikut adalah titik awal, bukan konfigurasi terbaik untuk semua dataset.

```python
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./results-indobert",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_emergency",
    greater_is_better=True,
    save_total_limit=2,
    logging_steps=10,
    report_to="none",
    seed=42
)
```

> Jika versi Transformers Anda tidak mengenali `eval_strategy`, periksa versi yang terpasang. Pada beberapa versi lama, nama argumennya adalah `evaluation_strategy`.

### Penjelasan parameter

| Parameter | Fungsi |
|---|---|
| `learning_rate=2e-5` | Langkah pembaruan bobot yang relatif kecil |
| `batch_size=8` | Jumlah contoh per batch per perangkat |
| `num_train_epochs=3` | Jumlah putaran training |
| `weight_decay=0.01` | Regularisasi untuk membantu mengurangi overfitting |
| `metric_for_best_model` | Metrik untuk memilih checkpoint terbaik |
| `load_best_model_at_end=True` | Memuat checkpoint terbaik setelah training |

## 13. Menjalankan Fine-Tuning

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

train_result = trainer.train()
print(train_result)
```

Contoh bentuk output:

```text
***** Running training *****
Num examples = 9
Num Epochs = 3
...
TrainOutput(
    global_step=6,
    training_loss=...,
    metrics={...}
)
```

Nilai loss dan metrik **tidak dicantumkan sebagai angka pasti** karena harus diperoleh dari eksekusi nyata. Dengan dataset sintetis yang sangat kecil, hasilnya tidak bermakna sebagai bukti kualitas model.

Jika versi Transformers Anda belum mendukung `processing_class`, ganti argumen itu dengan `tokenizer=tokenizer`.

## 14. Evaluasi pada Test Set

Lakukan evaluasi test setelah konfigurasi model dipilih berdasarkan validation set.

```python
test_metrics = trainer.evaluate(
    eval_dataset=tokenized_datasets["test"]
)

print(test_metrics)
```

Contoh bentuk output:

```text
{
    'eval_loss': ...,
    'eval_accuracy': ...,
    'eval_precision_emergency': ...,
    'eval_recall_emergency': ...,
    'eval_f1_emergency': ...
}
```

Angka harus berasal dari hasil eksekusi. Jangan mengisi tabel laporan dengan angka ilustrasi seolah-olah merupakan hasil eksperimen sungguhan.

### Confusion matrix dan classification report

```python
from sklearn.metrics import classification_report, confusion_matrix

pred_output = trainer.predict(tokenized_datasets["test"])
y_true = pred_output.label_ids
y_pred = np.argmax(pred_output.predictions, axis=-1)

print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["Non-Emergency", "Emergency"],
    zero_division=0
))
```

Contoh bentuk output confusion matrix:

```text
Confusion matrix:
[[TN FP]
 [FN TP]]
```

Keterangan:

- `TN`: Non-Emergency yang benar diprediksi Non-Emergency.
- `FP`: Non-Emergency yang keliru diprediksi Emergency.
- `FN`: Emergency yang keliru diprediksi Non-Emergency.
- `TP`: Emergency yang benar diprediksi Emergency.

Untuk penelitian bantuan darurat, analisis false negative secara khusus. Namun, sistem tetap harus divalidasi oleh manusia sebelum digunakan untuk keputusan operasional.

## 15. Menyimpan Model

```python
SAVE_DIR = "./indobert-emergency-model"

trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print("Model dan tokenizer disimpan ke:", SAVE_DIR)
```

Contoh output:

```text
Model dan tokenizer disimpan ke: ./indobert-emergency-model
```

Folder hasil penyimpanan biasanya berisi file konfigurasi, tokenizer, dan bobot model. Nama file bobot dapat berbeda bergantung pada format yang digunakan.

## 16. Prediksi Teks Baru

```python
import torch
from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model=SAVE_DIR,
    tokenizer=SAVE_DIR,
    device=0 if torch.cuda.is_available() else -1
)

texts = [
    "Tolong kirim bantuan, keluarga kami terjebak banjir.",
    "Pemerintah mengadakan sosialisasi kesiapsiagaan bencana."
]

predictions = classifier(texts, truncation=True)

for text, prediction in zip(texts, predictions):
    print("Teks:", text)
    print("Prediksi:", prediction)
    print()
```

Contoh bentuk output:

```text
Teks: Tolong kirim bantuan, keluarga kami terjebak banjir.
Prediksi: {'label': 'Emergency', 'score': ...}

Teks: Pemerintah mengadakan sosialisasi kesiapsiagaan bencana.
Prediksi: {'label': 'Non-Emergency', 'score': ...}
```

**Penting:** label dan skor di atas hanya menunjukkan bentuk output yang diharapkan, bukan hasil yang dijamin. Model yang dilatih pada data contoh yang kecil bisa salah memprediksi teks baru.

Skor `pipeline` adalah skor model untuk label yang dipilih, bukan otomatis probabilitas terkalibrasi. Untuk penggunaan nyata, evaluasi kalibrasi dan tetapkan threshold berdasarkan validation set.

## 17. Hal Penting untuk Penelitian

### 17.1 Buat baseline

Bandingkan IndoBERT dengan metode yang lebih sederhana:

1. TF-IDF + Logistic Regression.
2. TF-IDF + SVM.
3. IndoBERT fine-tuned.

Gunakan split data, label, dan prosedur evaluasi yang konsisten.

### 17.2 Hindari data leakage

- Jangan melakukan split setelah teks duplikat tersebar di semua subset.
- Fit TF-IDF hanya pada train set.
- Jangan memilih model berdasarkan test set.
- Dokumentasikan aturan preprocessing dan anotasi.

### 17.3 Periksa distribusi label

Jika kelas Emergency jauh lebih sedikit, laporkan distribusi kelas dan evaluasi precision, recall, F1, serta confusion matrix. Pertimbangkan class weighting atau strategi lain hanya berdasarkan kebutuhan yang terukur.

### 17.4 Analisis kesalahan

Periksa contoh:

- False negative: Emergency yang terlewat.
- False positive: teks umum yang dianggap Emergency.
- Teks ambigu, sarkasme, kutipan berita, dan laporan kejadian tanpa permintaan bantuan eksplisit.

### 17.5 Jangan menyamakan prediksi dengan prioritas operasional

Label Emergency dari model adalah alat bantu penyaringan, bukan bukti bahwa laporan benar, lokasi akurat, atau bantuan sudah dikirim. Sistem nyata memerlukan verifikasi, penanganan duplikasi, privasi, serta prosedur eskalasi manusia.

---

## 18. Latihan Mandiri

1. Tambahkan setidaknya 100 contoh berlabel yang telah diperiksa untuk latihan awal.
2. Buat pedoman anotasi Emergency dan Non-Emergency.
3. Ubah `max_length` menjadi 64 dan 128; bandingkan biaya dan hasilnya.
4. Uji `learning_rate` `2e-5` dan `3e-5` dengan validation set.
5. Catat precision, recall, F1, dan confusion matrix.
6. Bandingkan IndoBERT dengan TF-IDF + SVM.
7. Tinjau seluruh false negative untuk mencari pola kesalahan.
8. Simpan konfigurasi, seed, versi library, pembagian dataset, dan hasil eksperimen.

Jangan memilih konfigurasi berdasarkan satu kali eksperimen pada dataset kecil. Gunakan data yang memadai dan ulangi eksperimen bila diperlukan.

## 19. Checklist Penguasaan

- [ ] Saya dapat menjelaskan pre-training dan fine-tuning.
- [ ] Saya memahami tokenizer, input IDs, attention mask, padding, dan truncation.
- [ ] Saya dapat menyiapkan dataset teks dan label.
- [ ] Saya dapat membagi data menjadi train, validation, dan test tanpa kebocoran.
- [ ] Saya dapat memuat IndoBERT dan melakukan fine-tuning.
- [ ] Saya dapat menghitung precision, recall, F1, dan confusion matrix.
- [ ] Saya dapat menyimpan dan memuat kembali model.
- [ ] Saya dapat membandingkan IndoBERT dengan baseline TF-IDF.
- [ ] Saya dapat menganalisis false negative dan false positive.
- [ ] Saya memahami keterbatasan dataset contoh dan perlunya verifikasi manusia.

---

## 20. Ringkasan

Fine-tuning IndoBERT mengadaptasi model bahasa Indonesia pralatih untuk tugas tertentu dengan data berlabel. Untuk klasifikasi tweet permintaan bantuan darurat, tahapan utamanya adalah:

```text
Dataset berlabel
      ↓
Train / Validation / Test
      ↓
Tokenizer IndoBERT
      ↓
IndoBERT + Classification Head
      ↓
Fine-tuning
      ↓
Evaluasi pada Validation Set
      ↓
Pilih model terbaik
      ↓
Evaluasi Test Set
      ↓
Simpan model
      ↓
Prediksi / integrasi API
```

Kualitas penelitian tidak hanya ditentukan oleh arsitektur model. Kualitas label, representativitas dataset, pencegahan data leakage, pemilihan metrik, analisis kesalahan, dan pelaporan eksperimen sama pentingnya.

### Referensi untuk dibaca

- Hugging Face Transformers documentation: <https://huggingface.co/docs/transformers/>
- Hugging Face Trainer: <https://huggingface.co/docs/transformers/main_classes/trainer>
- Hugging Face Datasets: <https://huggingface.co/docs/datasets/>
- Model IndoBERT: <https://huggingface.co/indobenchmark/indobert-base-p1>
