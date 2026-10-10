# Evaluasi NLP (Natural Language Processing) untuk Klasifikasi Teks

## 1. Tujuan Pembelajaran

Setelah mempelajari materi ini, Anda diharapkan mampu:

- Menjelaskan mengapa model NLP perlu dievaluasi.
- Memahami pembagian data training, validation, dan testing.
- Membaca confusion matrix.
- Menghitung accuracy, precision, recall, dan F1-score.
- Memahami macro average dan weighted average.
- Mengevaluasi model ketika distribusi kelas tidak seimbang.
- Membandingkan model TF-IDF + Machine Learning dengan IndoBERT secara adil.
- Memilih metrik yang sesuai untuk penelitian deteksi permintaan bantuan darurat saat bencana.

## 2. Apa Itu Evaluasi NLP?

Evaluasi NLP adalah proses mengukur seberapa baik model memprediksi atau menyelesaikan tugas pemrosesan bahasa alami. Untuk tugas **text classification**, model menerima teks dan memprediksi label tertentu.

Contoh tugas penelitian:

- `Emergency`: teks menunjukkan permintaan bantuan atau kondisi darurat yang membutuhkan perhatian.
- `Non-Emergency`: teks tidak menunjukkan permintaan bantuan atau kondisi darurat sesuai pedoman anotasi penelitian.

Contoh:

| Teks | Label aktual |
|---|---|
| Tolong bantu kami, rumah terendam banjir | Emergency |
| Jalan menuju desa terputus akibat longsor | Emergency |
| Hujan turun sejak pagi di Kota Palu | Non-Emergency |
| Saya sedang membaca berita banjir | Non-Emergency |

Label aktual harus ditentukan berdasarkan **pedoman anotasi yang konsisten**, bukan sekadar keberadaan kata seperti “banjir” atau “tolong”.

Evaluasi membantu menjawab pertanyaan seperti:

1. Seberapa sering model membuat prediksi yang benar?
2. Seberapa banyak teks Emergency yang berhasil ditemukan?
3. Seberapa banyak prediksi Emergency ternyata salah?
4. Apakah model tetap baik ketika jumlah data antarkelas berbeda?
5. Apakah peningkatan model baru benar-benar lebih baik daripada baseline?

## 3. Pembagian Dataset

Sebelum melatih model, data umumnya dibagi menjadi tiga bagian.

### 3.1 Training set

Digunakan untuk melatih model agar mempelajari pola dari teks dan label.

### 3.2 Validation set

Digunakan untuk memilih hyperparameter, membandingkan konfigurasi, dan menentukan keputusan pengembangan model.

### 3.3 Test set

Digunakan untuk evaluasi akhir setelah model dan konfigurasi dipilih. Test set sebaiknya tidak digunakan berulang kali untuk mengambil keputusan selama pengembangan.

Contoh pembagian:

- Training: 70%
- Validation: 15%
- Testing: 15%

Proporsi tersebut bukan aturan mutlak. Untuk dataset yang kecil, pembagian 80:20 atau stratified cross-validation dapat lebih sesuai.

### 3.4 Stratified split

`Stratify` membantu menjaga proporsi label di setiap subset agar mendekati distribusi dataset awal.

```python
from sklearn.model_selection import train_test_split

X = [
    "Tolong bantu korban banjir",
    "Rumah terendam air",
    "Cuaca hari ini cerah",
    "Saya membaca berita banjir",
    "Kami membutuhkan makanan",
    "Saya sedang bekerja"
]

y = [
    "Emergency",
    "Emergency",
    "Non-Emergency",
    "Non-Emergency",
    "Emergency",
    "Non-Emergency"
]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y
)

print("Jumlah data train:", len(X_train))
print("Jumlah data test :", len(X_test))
print("Label train      :", y_train)
print("Label test       :", y_test)
```

Contoh keluaran:

```text
Jumlah data train: 4
Jumlah data test : 2
Label train      : ['Emergency', 'Non-Emergency', 'Emergency', 'Non-Emergency']
Label test       : ['Emergency', 'Non-Emergency']
```

**Catatan:** hasil split bergantung pada isi data, urutan data, versi library, dan parameter. Dataset mini di atas hanya untuk demonstrasi; jumlahnya terlalu kecil untuk evaluasi penelitian yang dapat dipercaya.

### 3.5 Hindari data leakage

Data leakage terjadi ketika informasi dari data evaluasi ikut memengaruhi proses pelatihan.

Untuk TF-IDF, jangan melakukan `fit_transform()` pada seluruh dataset sebelum split. Vectorizer harus mempelajari vocabulary dan bobot IDF dari data training saja.

Gunakan `Pipeline` agar proses ini lebih aman. Untuk IndoBERT, jangan gunakan test set untuk memilih epoch atau hyperparameter.

## 4. Confusion Matrix

Confusion matrix merangkum perbandingan label aktual dan prediksi model.

Untuk klasifikasi biner, kita menggunakan empat istilah.

| Istilah | Makna pada penelitian ini |
|---|---|
| True Positive (TP) | Emergency aktual diprediksi Emergency |
| True Negative (TN) | Non-Emergency aktual diprediksi Non-Emergency |
| False Positive (FP) | Non-Emergency aktual salah diprediksi Emergency |
| False Negative (FN) | Emergency aktual salah diprediksi Non-Emergency |

Contoh confusion matrix:

```text
                         Prediksi
                   Emergency  Non-Emergency
Aktual Emergency        8            2
Aktual Non-Emergency    3            7
```

Dari tabel tersebut:

- TP = 8
- FN = 2
- FP = 3
- TN = 7
- Total data = 20

Maknanya:

- Model menemukan 8 teks Emergency dengan benar.
- Model melewatkan 2 teks Emergency.
- Model memberi alarm Emergency yang salah pada 3 teks Non-Emergency.
- Model mengenali 7 teks Non-Emergency dengan benar.

Untuk memastikan urutan baris dan kolom, periksa dokumentasi fungsi yang digunakan. Pada `sklearn.metrics.confusion_matrix`, urutan label dapat ditentukan melalui parameter `labels`.

## 5. Accuracy

Accuracy mengukur proporsi seluruh prediksi yang benar.

\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\]

Dengan contoh sebelumnya:

\[
Accuracy = \frac{8 + 7}{8 + 7 + 3 + 2} = 0.75
\]

Hasilnya adalah **75%**.

Accuracy mudah dipahami, tetapi dapat menyesatkan jika kelas sangat tidak seimbang. Model yang selalu memilih kelas mayoritas mungkin mendapat accuracy tinggi walaupun gagal menemukan kelas penting.

## 6. Precision

Precision mengukur berapa banyak prediksi positif yang benar-benar positif.

\[
Precision = \frac{TP}{TP + FP}
\]

Contoh:

\[
Precision = \frac{8}{8 + 3} \approx 0.7273
\]

Hasilnya sekitar **72,73%**.

Artinya, dari semua teks yang diprediksi Emergency, sekitar 72,73% benar-benar Emergency menurut label aktual.

Precision penting ketika alarm palsu menimbulkan biaya, misalnya petugas perlu memeriksa laporan yang ternyata bukan keadaan darurat.

## 7. Recall

Recall mengukur berapa banyak data positif aktual yang berhasil ditemukan model.

\[
Recall = \frac{TP}{TP + FN}
\]

Contoh:

\[
Recall = \frac{8}{8 + 2} = 0.80
\]

Hasilnya **80%**.

Artinya, model menemukan 80% dari seluruh teks Emergency dalam data evaluasi.

Dalam penelitian deteksi permintaan bantuan darurat, **Recall kelas Emergency perlu mendapat perhatian khusus**, karena False Negative berarti laporan darurat yang ada dalam data tidak berhasil ditandai oleh model. Namun, recall tinggi tetap perlu dibaca bersama precision dan metrik lainnya.

## 8. F1-Score

F1-score adalah rata-rata harmonik precision dan recall.

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

Dengan precision sekitar 0,7273 dan recall 0,80:

\[
F1 \approx 0.7619
\]

Hasilnya sekitar **76,19%**.

F1 berguna ketika Anda ingin menilai precision dan recall secara bersamaan. F1 tidak memperhitungkan True Negative secara langsung dan tidak menggantikan pemeriksaan confusion matrix.

## 9. Menghitung Metrik dengan Scikit-learn

Jalankan contoh berikut. Data prediksi sengaja dibuat sederhana agar setiap komponen confusion matrix dapat diperiksa.

```python
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

y_true = [
    "Emergency", "Emergency", "Emergency", "Emergency",
    "Emergency", "Emergency", "Emergency", "Emergency",
    "Emergency", "Emergency",
    "Non-Emergency", "Non-Emergency", "Non-Emergency",
    "Non-Emergency", "Non-Emergency", "Non-Emergency",
    "Non-Emergency", "Non-Emergency", "Non-Emergency",
    "Non-Emergency"
]

y_pred = [
    "Emergency", "Emergency", "Emergency", "Emergency",
    "Emergency", "Emergency", "Emergency", "Emergency",
    "Non-Emergency", "Non-Emergency",
    "Emergency", "Emergency", "Emergency",
    "Non-Emergency", "Non-Emergency", "Non-Emergency",
    "Non-Emergency", "Non-Emergency", "Non-Emergency",
    "Non-Emergency"
]

labels = ["Emergency", "Non-Emergency"]

cm = confusion_matrix(y_true, y_pred, labels=labels)

print("Confusion Matrix:")
print(cm)
print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
print("Precision:", round(
    precision_score(y_true, y_pred, pos_label="Emergency"), 4
))
print("Recall   :", round(
    recall_score(y_true, y_pred, pos_label="Emergency"), 4
))
print("F1-score :", round(
    f1_score(y_true, y_pred, pos_label="Emergency"), 4
))
print("\\nClassification Report:")
print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
```

Output utama:

```text
Confusion Matrix:
[[ 8  2]
 [ 3  7]]

Accuracy : 0.75
Precision: 0.7273
Recall   : 0.8
F1-score : 0.7619
```

Laporan klasifikasi kira-kira menampilkan nilai berikut (pembulatan dapat berbeda menurut versi Scikit-learn):

```text
               precision    recall  f1-score   support

    Emergency       0.73      0.80      0.76        10
Non-Emergency       0.78      0.70      0.74        10

     accuracy                           0.75        20
    macro avg       0.75      0.75      0.75        20
 weighted avg       0.75      0.75      0.75        20
```

## 10. Macro Average dan Weighted Average

Ketika terdapat lebih dari dua kelas atau ingin merangkum performa tiap kelas, perhatikan cara rata-rata dihitung.

### 10.1 Macro average

Menghitung metrik untuk setiap kelas, lalu merata-ratakannya tanpa memperhatikan jumlah data tiap kelas.

- Setiap kelas memiliki bobot yang sama.
- Berguna ketika performa kelas minoritas sama pentingnya dengan kelas mayoritas.

### 10.2 Weighted average

Merata-ratakan metrik setiap kelas dengan bobot sesuai jumlah data aktual (*support*) kelas tersebut.

- Kelas dengan data lebih banyak memiliki pengaruh lebih besar.
- Bisa terlihat tinggi ketika performa pada kelas mayoritas bagus, meskipun kelas minoritas kurang baik.

### 10.3 Micro average

Menggabungkan kontribusi TP, FP, dan FN dari semua kelas sebelum menghitung metrik. Pada klasifikasi single-label multiclass, micro F1 biasanya sama dengan accuracy.

Untuk penelitian Emergency/Non-Emergency, laporkan setidaknya precision, recall, dan F1 per kelas serta macro F1. Jangan hanya mengandalkan weighted F1.

## 11. Masalah Class Imbalance

Misalnya dataset terdiri dari:

```text
Emergency       : 100 data
Non-Emergency   : 900 data
Total           : 1.000 data
```

Model sederhana yang selalu memprediksi `Non-Emergency` akan mendapatkan accuracy 90%, tetapi recall Emergency = 0%.

Contoh:

```python
from sklearn.metrics import accuracy_score, recall_score, f1_score

y_true = ["Emergency"] * 100 + ["Non-Emergency"] * 900
y_pred = ["Non-Emergency"] * 1000

print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
print("Recall Emergency:", round(
    recall_score(y_true, y_pred, pos_label="Emergency", zero_division=0), 4
))
print("F1 Emergency:", round(
    f1_score(y_true, y_pred, pos_label="Emergency", zero_division=0), 4
))
```

Output:

```text
Accuracy: 0.9
Recall Emergency: 0.0
F1 Emergency: 0.0
```

Contoh ini menunjukkan mengapa accuracy saja tidak cukup.

### 11.1 Strategi penanganan

Beberapa pendekatan yang bisa diuji:

- Mengumpulkan lebih banyak contoh kelas minoritas.
- Menggunakan `class_weight="balanced"` pada model yang mendukungnya.
- Melakukan oversampling atau undersampling hanya pada data training.
- Menggunakan weighted loss untuk model neural network.
- Menetapkan threshold berdasarkan validation set, jika model menghasilkan probabilitas.
- Melaporkan precision, recall, F1 per kelas, macro F1, dan confusion matrix.

Jangan melakukan oversampling sebelum split karena dapat menyebabkan data duplikat atau sangat mirip tersebar ke training dan test.

## 12. Cross-Validation

Cross-validation menilai model melalui beberapa pembagian data training, bukan hanya satu pembagian.

Contoh 5-fold cross-validation:

```text
Fold 1: validasi pada bagian 1, training pada bagian lainnya
Fold 2: validasi pada bagian 2, training pada bagian lainnya
Fold 3: validasi pada bagian 3, training pada bagian lainnya
Fold 4: validasi pada bagian 4, training pada bagian lainnya
Fold 5: validasi pada bagian 5, training pada bagian lainnya
```

Contoh penggunaan untuk pipeline TF-IDF + SVM:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer

texts = [
    "tolong bantu korban banjir",
    "rumah saya terendam banjir",
    "kami butuh makanan",
    "jalan desa terputus longsor",
    "cuaca hari ini cerah",
    "saya membaca berita banjir",
    "sedang makan siang",
    "langit terlihat biru",
    "tolong evakuasi warga",
    "informasi cuaca sore ini"
]

labels = [
    "Emergency", "Emergency", "Emergency", "Emergency",
    "Non-Emergency", "Non-Emergency", "Non-Emergency",
    "Non-Emergency", "Emergency", "Non-Emergency"
]

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", LinearSVC(class_weight="balanced"))
])

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

scores = cross_validate(
    pipeline,
    texts,
    labels,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "precision_emergency": make_scorer(
            precision_score, pos_label="Emergency", zero_division=0
        ),
        "recall_emergency": make_scorer(
            recall_score, pos_label="Emergency", zero_division=0
        ),
        "f1_emergency": make_scorer(
            f1_score, pos_label="Emergency", zero_division=0
        )
    },
    return_train_score=False
)

for metric in [
    "test_accuracy",
    "test_precision_emergency",
    "test_recall_emergency",
    "test_f1_emergency"
]:
    print(metric, round(scores[metric].mean(), 4))
```

**Penting:** contoh ini menetapkan `pos_label="Emergency"` secara eksplisit agar precision, recall, dan F1 yang dihitung memang untuk kelas Emergency. Dataset mini ini hanya demonstrasi dan terlalu kecil untuk menghasilkan estimasi performa yang stabil. Untuk evaluasi penelitian, gunakan data yang lebih besar dan cukup banyak contoh pada setiap kelas di tiap fold.

## 13. ROC-AUC dan Precision-Recall AUC

### 13.1 ROC-AUC

ROC-AUC mengukur kemampuan model memeringkat contoh positif di atas contoh negatif pada berbagai threshold. Perhitungan memerlukan skor kontinu, misalnya probabilitas atau decision score, bukan hanya label prediksi.

### 13.2 Precision-Recall AUC

Kurva Precision-Recall sering lebih informatif ketika kelas positif jarang, seperti Emergency pada dataset yang didominasi Non-Emergency.

Pada situasi class imbalance, pertimbangkan untuk melaporkan PR-AUC bersama recall, precision, F1, dan confusion matrix. Pastikan skor yang digunakan memang berasal dari model dan kelas positif didefinisikan dengan benar.

## 14. Evaluasi Model TF-IDF dan IndoBERT secara Adil

Untuk penelitian, Anda dapat membandingkan:

1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM
3. IndoBERT fine-tuning

Agar perbandingan valid:

- Gunakan data test yang sama untuk semua model.
- Pisahkan data sebelum fitting vectorizer atau melakukan resampling.
- Gunakan aturan anotasi dan label yang sama.
- Tentukan metrik utama sebelum melihat hasil test.
- Pilih hyperparameter dan checkpoint menggunakan data validation, bukan test.
- Catat random seed, versi library, parameter, dan waktu pelatihan.
- Laporkan hasil per kelas, macro F1, serta Recall Emergency.
- Jika memungkinkan, gunakan beberapa seed atau cross-validation pada data pengembangan.
- Periksa kemungkinan tweet duplikat, retweet, atau tweet dari kejadian yang sama tersebar di train dan test.

Jika tujuan penggunaan adalah memprioritaskan laporan bencana baru, pertimbangkan split berdasarkan waktu atau kejadian bencana sebagai pengujian tambahan. Random split dapat memberi hasil terlalu optimistis ketika teks yang sangat mirip berada pada train dan test.

### Template tabel hasil

| Model | Accuracy | Precision Emergency | Recall Emergency | F1 Emergency | Macro F1 |
|---|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | — | — | — | — | — |
| TF-IDF + Linear SVM | — | — | — | — | — |
| IndoBERT | — | — | — | — | — |

Isi tabel dengan hasil eksperimen sebenarnya. Jangan mengisi angka sebelum model diuji pada data yang ditetapkan.

## 15. Memilih Metrik Utama untuk Penelitian

Untuk deteksi tweet permintaan bantuan darurat:

- **Recall Emergency**: mengukur seberapa banyak laporan darurat aktual yang berhasil ditemukan.
- **Precision Emergency**: mengukur seberapa banyak laporan yang ditandai darurat ternyata benar.
- **Macro F1**: menilai keseimbangan performa antarkelas.
- **Confusion matrix**: menunjukkan jenis kesalahan model.
- **Accuracy**: tetap berguna sebagai metrik pelengkap, bukan satu-satunya dasar kesimpulan.

Tidak ada satu metrik yang cukup untuk semua kondisi. Recall tinggi dapat disertai banyak false positive; karena itu, pertimbangkan juga kapasitas petugas untuk meninjau laporan dan konsekuensi dari tiap jenis kesalahan.

## 16. Kesalahan Umum dalam Evaluasi NLP

1. Hanya melaporkan accuracy.
2. Menggunakan test set untuk memilih hyperparameter.
3. Melakukan `fit_transform()` TF-IDF sebelum split.
4. Melakukan oversampling pada seluruh dataset sebelum split.
5. Mengabaikan class imbalance.
6. Tidak menjelaskan kelas positif.
7. Membandingkan model pada test set yang berbeda.
8. Menganggap confidence 0,95 otomatis berarti prediksi benar dengan probabilitas 95% tanpa kalibrasi.
9. Mengabaikan tweet duplikat atau kebocoran informasi antarpartisi.
10. Menyimpulkan model siap digunakan di dunia nyata hanya berdasarkan hasil test internal.

## 17. Latihan

Gunakan confusion matrix berikut:

```text
                         Prediksi
                   Emergency  Non-Emergency
Aktual Emergency        45           15
Aktual Non-Emergency    10           130
```

Jawab pertanyaan berikut:

1. Berapa TP, TN, FP, dan FN?
2. Hitung accuracy.
3. Hitung precision Emergency.
4. Hitung recall Emergency.
5. Hitung F1-score Emergency.
6. Menurut Anda, kesalahan mana yang lebih berisiko untuk sistem prioritas bantuan bencana: FP atau FN? Jelaskan bahwa jawabannya bergantung pada konteks operasional.

### Kunci jawaban

- TP = 45
- FN = 15
- FP = 10
- TN = 130
- Total = 200
- Accuracy = (45 + 130) / 200 = 0,875 atau 87,5%
- Precision Emergency = 45 / (45 + 10) ≈ 0,8182 atau 81,82%
- Recall Emergency = 45 / (45 + 15) = 0,75 atau 75%
- F1 Emergency ≈ 0,7826 atau 78,26%

## 18. Checklist Penguasaan

- [ ] Saya memahami train, validation, dan test set.
- [ ] Saya bisa membaca confusion matrix.
- [ ] Saya bisa menjelaskan TP, TN, FP, dan FN.
- [ ] Saya bisa menghitung accuracy, precision, recall, dan F1.
- [ ] Saya memahami macro dan weighted average.
- [ ] Saya memahami mengapa accuracy bisa menyesatkan pada class imbalance.
- [ ] Saya bisa mengevaluasi Recall Emergency secara eksplisit.
- [ ] Saya memahami data leakage pada TF-IDF dan resampling.
- [ ] Saya bisa membandingkan TF-IDF + SVM dengan IndoBERT secara adil.
- [ ] Saya bisa menjelaskan keterbatasan hasil evaluasi sebelum deployment.

## 19. Ringkasan

Evaluasi adalah bagian inti dari penelitian NLP, bukan sekadar langkah setelah training. Untuk klasifikasi teks, kuasai confusion matrix, precision, recall, F1-score, pembagian dataset, class imbalance, dan pencegahan data leakage.

Dalam penelitian deteksi permintaan bantuan darurat, perhatikan terutama **Recall Emergency**, tetapi selalu laporkan precision, F1, macro F1, dan confusion matrix agar trade-off model terlihat. Gunakan TF-IDF + Logistic Regression/SVM sebagai baseline dan bandingkan dengan IndoBERT pada protokol evaluasi yang sama.
