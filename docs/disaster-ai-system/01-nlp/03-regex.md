# Regular Expression (Regex) untuk Penelitian NLP Berbasis Media Sosial

Regular Expression (Regex) adalah bahasa pola untuk mencari,
mencocokkan, mengekstraksi, memvalidasi, atau mengganti bagian teks
berdasarkan aturan tertentu. Dalam penelitian Natural Language
Processing (NLP), Regex sering digunakan pada tahap *text preprocessing*
dan ekstraksi informasi sebelum teks diproses oleh metode statistik,
machine learning, atau deep learning.

Materi ini dirancang untuk mendukung penelitian NLP berbasis teks media
sosial, khususnya pengolahan unggahan terkait bencana atau permintaan
bantuan darurat. Contoh menggunakan Python dan modul bawaan `re`,
sehingga dapat dipraktikkan di Jupyter Notebook, Google Colab, atau VS
Code.

Regex bukan alat untuk memahami makna bahasa secara menyeluruh. Regex
cocok untuk pola eksplisit, seperti URL, mention, hashtag, nomor dengan
format tertentu, pengulangan tanda baca, atau kata kunci yang telah
ditentukan. Untuk memahami konteks, urgensi, sentimen, atau maksud
unggahan, Regex sebaiknya dikombinasikan dengan anotasi data dan model
NLP.

## Tujuan Pembelajaran

Setelah mempelajari materi ini, peneliti diharapkan mampu:

1.  Memahami sintaks dan cara kerja Regex.
2.  Menulis pola untuk menemukan kata, angka, simbol, dan pola teks.
3.  Menggunakan Regex melalui fungsi Python `re`.
4.  Membersihkan teks media sosial dengan aturan yang transparan dan
    dapat direproduksi.
5.  Mengekstraksi fitur eksplisit seperti URL, hashtag, mention, dan
    pola nomor.
6.  Menyusun pipeline preprocessing yang dapat diuji.
7.  Menilai risiko hilangnya informasi akibat pembersihan teks.
8.  Mendokumentasikan aturan Regex agar eksperimen dapat direplikasi.

## Posisi Regex dalam Pipeline Penelitian NLP

Pipeline penelitian yang umum:

1.  Pengumpulan data sesuai ketentuan platform, etika penelitian, dan
    kebijakan privasi.
2.  Penyimpanan data mentah secara aman.
3.  Pemeriksaan kualitas dan penghapusan duplikasi berdasarkan aturan
    yang ditetapkan.
4.  Eksplorasi data awal (*Exploratory Data Analysis*).
5.  Preprocessing menggunakan aturan terdokumentasi, termasuk Regex bila
    relevan.
6.  Anotasi label dengan prosedur yang dapat dipertanggungjawabkan.
7.  Pembagian data train, validation, dan test tanpa kebocoran data.
8.  Ekstraksi fitur atau tokenisasi sesuai model.
9.  Pelatihan dan evaluasi model.
10. Analisis kesalahan serta pelaporan keterbatasan.

Urutan beberapa langkah dapat berbeda menurut desain penelitian. Simpan
teks mentah dan teks hasil preprocessing secara terpisah agar keputusan
pembersihan dapat diaudit.

### Contoh data mentah

``` text
URGENT!!! Banjir di Desa Contoh 😥 air naik cepat.
Mohon bantuan evakuasi keluarga! #Banjir
Info: https://contoh.id/laporan @relawan
```

### Contoh hasil preprocessing konservatif

``` text
URGENT! Banjir di Desa Contoh 😥 air naik cepat.
Mohon bantuan evakuasi keluarga! #Banjir
Info: @relawan
```

Ini hanya ilustrasi. URL dihapus karena dianggap tidak diperlukan untuk
konfigurasi model tertentu, sedangkan hashtag, emoji, dan kata `URGENT`
dipertahankan karena mungkin mengandung sinyal penting. Keputusan harus
diuji, bukan dianggap selalu benar.

## Persiapan Lingkungan Python

Modul `re` merupakan modul bawaan Python.

``` python
import re

text = "Saya belajar NLP pada tahun 2026."
match = re.search(r"nlp", text, flags=re.IGNORECASE)
print(match.group() if match else "Tidak ditemukan")
```

Gunakan *raw string* seperti `r"\d+"` ketika menulis pola Regex di
Python. Raw string membantu menghindari kebingungan antara escape
sequence Python dan Regex.

## Fundamental Regex

### Literal Characters

Literal adalah karakter yang dicocokkan secara langsung.

``` python
import re

text = "Laporan banjir diterima."
print(re.findall(r"banjir", text))
```

Output:

``` text
['banjir']
```

Pencocokan secara default membedakan huruf besar dan kecil. Gunakan
`re.IGNORECASE` jika kapitalisasi tidak penting.

### Metacharacters

| Simbol | Fungsi                                                          | Contoh           |
|:-------|:----------------------------------------------------------------|:-----------------|
| `.`    | Satu karakter apa pun selain newline secara default             | `b.n`            |
| `^`    | Awal string; dengan mode multiline dapat berarti awal baris     | `^Mohon`         |
| `$`    | Akhir string; dengan mode multiline dapat berarti akhir baris   | `darurat$`       |
| `*`    | Nol kali atau lebih                                             | `ab*`            |
| `+`    | Satu kali atau lebih                                            | `ab+`            |
| `?`    | Nol atau satu kali; juga dapat membuat quantifier bersifat lazy | `colou?r`        |
| `[]`   | Salah satu karakter di dalam kelas                              | `[abc]`          |
| `()`   | Grup pencocokan                                                 | `(banjir|gempa)` |
| `|`    | Alternatif                                                      | `banjir|gempa`   |
| `\`    | Escape atau pembentuk kelas khusus                              | `\.`             |

Untuk mencari titik literal, gunakan `\.` karena `.` tanpa escape
memiliki makna khusus.

``` python
text = "Berkas laporan.txt dan laporanxt"
print(re.findall(r"laporan\.txt", text))
```

### Character Classes

| Pola     | Makna                                               |
|:---------|:----------------------------------------------------|
| `[abc]`  | Salah satu dari `a`, `b`, atau `c`                  |
| `[a-z]`  | Huruf kecil ASCII dari `a` sampai `z`               |
| `[A-Z]`  | Huruf besar ASCII dari `A` sampai `Z`               |
| `[0-9]`  | Satu digit ASCII                                    |
| `[^0-9]` | Satu karakter yang bukan digit ASCII                |
| `\d`     | Digit menurut perilaku Unicode Python               |
| `\w`     | Karakter kata, umumnya huruf, digit, dan underscore |
| `\s`     | Karakter whitespace                                 |
| `\D`     | Bukan digit                                         |
| `\W`     | Bukan karakter kata                                 |
| `\S`     | Bukan whitespace                                    |

Rentang `[a-z]` tidak mencakup semua huruf beraksen atau seluruh
karakter Unicode. Hindari asumsi bahwa rentang ASCII mewakili semua
huruf.

### Quantifiers

| Pola     | Makna                     |
|:---------|:--------------------------|
| `a*`     | Nol atau lebih huruf `a`  |
| `a+`     | Satu atau lebih huruf `a` |
| `a?`     | Nol atau satu huruf `a`   |
| `a{3}`   | Tepat tiga huruf `a`      |
| `a{2,5}` | Dua sampai lima huruf `a` |
| `a{2,}`  | Dua atau lebih huruf `a`  |

Contoh mencari kelompok digit:

``` python
text = "Ada 3 korban, 12 rumah, dan 105 paket bantuan."
numbers = re.findall(r"\d+", text)
print(numbers)
```

Output:

``` text
['3', '12', '105']
```

Regex menemukan rangkaian digit, tetapi tidak mengetahui apakah angka
berarti jumlah korban, rumah, usia, tahun, atau nomor laporan.
Interpretasi membutuhkan konteks.

### Anchors dan Word Boundaries

Anchor mencocokkan posisi, bukan karakter biasa.

``` python
texts = [
    "Banjir melanda desa.",
    "Laporan: banjir melanda desa.",
    "banjir perlu ditangani"
]

for text in texts:
    print(bool(re.search(r"^banjir", text, flags=re.IGNORECASE)))
```

Output:

```text
True
False
True
```

`\b` mencocokkan batas kata menurut definisi karakter kata Regex.

``` python
text = "banjir, banjiran, dan antisipasi banjir"
print(re.findall(r"\bbanjir\b", text, flags=re.IGNORECASE))
```

Output:

```text
['banjir', 'banjir']
```

Batas kata Regex tidak selalu sama dengan batas kata linguistik. Uji
pola dengan contoh nyata dari korpus penelitian.

## Alternation, Groups, dan Capturing

### Alternation

Alternation menggunakan `|` untuk memilih salah satu pola.

``` python
pattern = r"\b(banjir|gempa|longsor)\b"
text = "Gempa memicu longsor, sementara wilayah lain mengalami banjir."
print(re.findall(pattern, text, flags=re.IGNORECASE))
```

Output:

```text
['Gempa', 'longsor', 'banjir']
```

Karena pola memiliki capturing group, `findall()` mengembalikan isi
grup. Jika tidak membutuhkan isi grup secara terpisah, gunakan grup
non-capturing `(?:...)`.

``` python
pattern = r"\b(?:banjir|gempa|longsor)\b"
```

Daftar kata kunci sebaiknya disusun berdasarkan kajian literatur,
karakteristik data, dan validasi manusia. Daftar kata kunci bukan
detektor bencana yang lengkap.

### Capturing Groups

``` python
text = "Kode laporan: EVT-2026-0042"
match = re.search(r"(EVT)-(\d{4})-(\d+)", text)

if match:
    print(match.group(0))  # seluruh pola
    print(match.group(1))  # EVT
    print(match.group(2))  # 2026
    print(match.group(3))  # 0042
```

Output:

```text
EVT-2026-0042
EVT
2026
0042
```

Grup bernama (berlabel) dapat membuat ekstraksi lebih mudah dibaca.

``` python
pattern = r"(?P<prefix>[A-Z]+)-(?P<year>\d{4})-(?P<number>\d+)"
match = re.search(pattern, "EVT-2026-0042")

if match:
    print(match.groupdict())
```

Output:

```text
{'prefix': 'EVT', 'year': '2026', 'number': '0042'}
```

Contoh tersebut hanya sesuai jika kode laporan memang mengikuti format
yang ditentukan.

## Greedy dan Lazy Matching

Quantifier seperti `*` dan `+` secara default bersifat greedy: pola
mencoba mengambil bagian sepanjang mungkin selama masih menghasilkan
kecocokan.

``` python
text = "<post>banjir</post><post>gempa</post>"

print(re.findall(r"<post>.*</post>", text))
print(re.findall(r"<post>.*?</post>", text))
```

Output:

```text
['<post>banjir</post><post>gempa</post>']
['<post>banjir</post>', '<post>gempa</post>']
```

Pola pertama dapat mencocokkan dari tag pembuka pertama hingga tag
penutup terakhir. Pola kedua memakai `.*?` agar mencocokkan sesingkat
mungkin.

Untuk HTML, XML, atau JSON kompleks, gunakan parser khusus daripada
Regex sebagai satu-satunya alat.

!!! info "Info"
    Karakter `?` (tanda tanya) dalam Regular Expression (Regex) memiliki beberapa fungsi tergantung bagaimana dan di mana ia digunakan.

## Lookahead dan Lookbehind

Lookaround memeriksa konteks di sekitar pola tanpa memasukkan konteks
tersebut ke hasil kecocokan.

### Positive Lookahead

*Lookahead Positive* `(?=...)` memeriksa apakah teks di depan cocok dengan pola tanpa memasukkannya ke hasil pencocokan.

``` python
text = "ID-123 ID-456 ID-AB7"
pattern = r"ID-(?=\d{3}\b)\w+"
print(re.findall(pattern, text))
```

Output:

```text
['ID-123', 'ID-456']
```

### Negative Lookahead

*Lookahead Negative* `(?!...)` memeriksa agar teks di depan tidak diikuti oleh pola tertentu. 
Misalnya ingin mencari kata "banjir", tetapi menolak jika kata tersebut diikuti oleh kata "bandang".

``` python
pattern = r"\bbanjir\b(?! bandang)"
```

Pola tersebut mengecualikan kecocokan `banjir` yang langsung diikuti
teks `bandang`. Contoh ini ilustratif; pengecualian dapat menghapus
konteks penting sehingga harus divalidasi.

### Lookbehind

*Lookbehind* `(?<=...)` atau `(?<!...)` memeriksa konteks karakter yang ada di sebelah kiri/sebelum pola.

``` python
text = "Korban: 12 orang; Rumah: 5 unit"
pattern = r"(?<=Korban: )\d+"
print(re.findall(pattern, text))
```

Lookbehind pada Python `re` memiliki batasan tertentu, termasuk
persyaratan panjang pola yang tetap. Untuk pola rumit, pilih desain
ekstraksi yang mudah diuji.

## Fungsi Regex Penting dalam Python

### `re.search()`

Mencari kecocokan pertama di mana pun dalam string.

``` python
re.search(r"banjir", "Wilayah ini terdampak banjir")
```

### `re.match()`

Memeriksa kecocokan mulai dari awal string.

``` python
re.match(r"Wilayah", "Wilayah ini terdampak banjir")
```

### `re.fullmatch()`

Memastikan seluruh string cocok dengan pola. Berguna untuk validasi
format, bukan untuk memastikan kebenaran semantik.

``` python
code = "EVT-2026-0042"
valid = re.fullmatch(r"EVT-\d{4}-\d{4}", code) is not None
print(valid)
```

Output:

```text
True
```

### `re.findall()`

Mengembalikan semua kecocokan. Bentuk hasil bergantung pada jumlah
capturing group.

``` python
re.findall(r"#\w+", "Banjir #Siaga #Bantuan")
```

### `re.finditer()`

Mengembalikan iterator objek kecocokan beserta posisi karakter.

``` python
text = "Banjir di desa. Gempa di kota."
for match in re.finditer(r"\b(banjir|gempa)\b", text, re.IGNORECASE):
    print(match.group(), match.start(), match.end())
```

Output:

```text
Banjir 0 6
Gempa 16 21
```

### `re.sub()`

Mengganti bagian yang cocok.

``` python
text = "Butuh bantuan!!!!"
clean = re.sub(r"!{2,}", "!", text)
print(clean)
```

Output:

```text
Butuh bantuan!
```

### `re.split()`

Memisahkan string berdasarkan pola.

``` python
text = "banjir, gempa; longsor"
parts = re.split(r"[,;]\s*", text)
print(parts)
```

Output:

```text
['banjir', 'gempa', 'longsor']
```

### `re.compile()`

Menyimpan pola untuk digunakan berulang kali.

``` python
keyword_pattern = re.compile(
    r"\b(?:banjir|gempa|longsor)\b",
    flags=re.IGNORECASE
)

for text in ["Ada banjir", "Terjadi gempa", "Cuaca cerah"]:
    print(bool(keyword_pattern.search(text)))
```

Output:

```text
True
True
False
```

## Flags yang Sering Digunakan

| Flag                        | Fungsi                                                                                               |
|:----------------------------|:-----------------------------------------------------------------------------------------------------|
| `re.IGNORECASE` atau `re.I` | Mengabaikan perbedaan huruf besar dan kecil                                                          |
| `re.MULTILINE` atau `re.M`  | Membuat `^` dan `$` bekerja pada awal/akhir setiap baris                                             |
| `re.DOTALL` atau `re.S`     | Membuat `.` juga cocok dengan newline                                                                |
| `re.VERBOSE` atau `re.X`    | Memungkinkan pola multi-baris dengan komentar dan spasi yang diabaikan di sebagian besar bagian pola |

Contoh pola verbose:

``` python
pattern = re.compile(
    r"""
    \b
    (?:banjir|gempa|longsor)
    \b
    """,
    flags=re.IGNORECASE | re.VERBOSE
)
```

Gunakan `re.VERBOSE` untuk pola panjang agar lebih mudah dibaca dan
ditinjau peneliti lain.

## Penggunaan `?`

Berikut adalah tabel ringkasan penggunaan karakter **`?`** dalam Regular Expression (Regex):

| Sintaks / Penggunaan | Jenis / Kategori | Deskripsi / Cara Kerja | Contoh Pola | Contoh Cocok (*Match*) |
| --- | --- | --- | --- | --- |
| **`a?`** | Quantifier (*Optional*) | Mencocokkan karakter atau grup sebanyak **0 atau 1 kali** (sifatnya opsional). | `https?://` | `"http://"` dan `"https://"` |
| **`*?`** atau **`+?`** | Non-Greedy / *Lazy* | Memaksa quantifier (`*` atau `+`) mengambil teks **sesingkat mungkin**. | `<post>.*?</post>` | `<post>banjir</post>` (berhenti di tag pertama) |
| **`(?:...)`** | Non-Capturing Group | Mengelompokkan pola **tanpa menyimpan** hasilnya ke memori/grup tangkapan. | `\b(?:banjir|gempa)\b` | Mengelompokkan kata tanpa membuat *capture group*. |
| **`(?P<nama>...)`** | Named Capturing Group | Memberi **nama identifikasi** pada grup tangkapan agar mudah dipanggil. | `(?P<year>\d{4})` | Mengambil `2026` dengan kunci `'year'`. |
| **`(?=...)`** | Positive Lookahead | Memeriksa apakah teks di depan **sesuai** pola, tanpa memasukkannya ke hasil pencocokan. | `ID-(?=\d{3})` | `ID-` pada `"ID-123"` (hanya mengambil `'ID-'`). |
| **`(?!...)`** | Negative Lookahead | Memeriksa agar teks di depan **TIDAK diikuti** oleh pola tertentu. | `\bbanjir\b(?! bandang)` | `"banjir"` pada `"banjir lokal"` (menolak `"banjir bandang"`). |
| **`(?<=...)`** | Positive Lookbehind | Memeriksa apakah teks di sebelah kiri/sebelumnya **sesuai** pola. | `(?<=Korban: )\d+` | `12` pada `"Korban: 12 orang"` (hanya mengambil angka). |
| **`(?<!...)`** | Negative Lookbehind | Memeriksa agar teks di sebelah kiri/sebelumnya **TIDAK diikuti** pola tertentu. | `(?<!\w)#[A-Za-z0-9_]+` | `#Banjir` (memastikan `#` tidak menempel di huruf lain). |

## Penerapan Regex untuk Data Media Sosial

### Menghapus URL

``` python
text = "Laporan tersedia di https://contoh.id/laporan dan http://contoh.id"
url_pattern = r"https?://\S+"
print(re.sub(url_pattern, "", text))
```

Output:

```text
Laporan tersedia di  dan 
```

Pola sederhana ini belum menangani seluruh kemungkinan URL. URL dapat
memiliki tanda baca di bagian akhir, karakter Unicode, atau format tanpa
skema `http`. Tentukan aturan sesuai kebutuhan dataset dan simpan URL
asli jika diperlukan untuk audit.

### Mengekstraksi URL

``` python
urls = re.findall(r"https?://\S+", text)
```

Ekstraksi URL tidak berarti peneliti harus membuka atau mengunjungi
tautan tersebut. Ikuti aturan keamanan, etika, dan kebijakan pengumpulan
data.

### Mengekstraksi Hashtag

``` python
text = "Waspada #Banjir dan #InfoBencana2026"
hashtags = re.findall(r"(?<!\w)#[A-Za-z0-9_]+", text)
print(hashtags)
```

Output:

```text
['#Banjir', '#InfoBencana2026']
```

Pola ini terutama mengakomodasi huruf ASCII, angka, dan underscore.
Untuk hashtag berkarakter Unicode, sesuaikan implementasi dan uji pada
data nyata. Hashtag dapat menjadi fitur penting, jadi jangan
menghapusnya sebelum menentukan tujuan analisis.

### Mengekstraksi Mention

``` python
text = "Mohon bantuan @relawan_palu dan @petugas"
mentions = re.findall(r"(?<!\w)@[\w]+", text)
print(mentions)
```

Output:

```text
['@relawan_palu', '@petugas']
```

Mention dapat mengandung informasi identitas. Pertimbangkan anonimisasi,
minimisasi data, dan aturan platform sebelum menyimpan atau
memublikasikannya.

### Mengurangi Tanda Baca Berulang

``` python
text = "Tolong bantu!!! Air naik cepat????"
clean = re.sub(r"([!?.,])\1+", r"\1", text)
print(clean)
```

Output:

```text
Tolong bantu! Air naik cepat?
```

Jangan langsung menghapus semua tanda baca. Banyaknya tanda seru atau
tanda tanya mungkin menjadi sinyal intensitas bahasa. Uji apakah
normalisasi tersebut memengaruhi hasil model.

### Merapikan Whitespace

``` python
text = "  Mohon   bantuan\n\nuntuk warga.  "
clean = re.sub(r"\s+", " ", text).strip()
print(clean)
```

Output:

```text
Mohon bantuan untuk warga.
```

### Menghapus Karakter Kontrol Tertentu

``` python
text = "Mohon bantuan\tuntuk warga\nterdampak."
clean = re.sub(r"[\t\r\n]+", " ", text)
print(clean)
```

Output:

```text
Mohon bantuan untuk warga terdampak.
```

Jangan menghapus seluruh karakter non-ASCII tanpa alasan. Cara tersebut
berisiko menghilangkan emoji, huruf non-Latin, atau informasi bermakna.

## Merancang Fungsi Preprocessing yang Transparan

Fungsi preprocessing harus mudah dibaca, diuji, dan disesuaikan. Contoh
berikut konservatif: menghapus URL, merapikan whitespace, dan mengurangi
tanda baca berulang. Fungsi ini tidak menghapus hashtag, mention, angka,
emoji, atau kata kunci.

``` python
import re

URL_PATTERN = re.compile(r"https?://\S+", flags=re.IGNORECASE)
REPEATED_PUNCTUATION = re.compile(r"([!?.,])\1+")
WHITESPACE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    """Membersihkan beberapa pola eksplisit dari teks."""
    if not isinstance(text, str):
        return ""

    text = URL_PATTERN.sub(" ", text)
    text = REPEATED_PUNCTUATION.sub(r"\1", text)
    text = WHITESPACE.sub(" ", text).strip()
    return text

examples = [
    "URGENT!!! Banjir di desa 😥",
    "Mohon bantuan https://contoh.id/laporan",
    None,
]

for item in examples:
    print(clean_text(item))
```

Output:

```
URGENT! Banjir di desa 😥
Mohon bantuan
```

Sebelum menggunakan fungsi dalam penelitian, dokumentasikan alasan
setiap aturan, periksa hasil pada sampel, dan bandingkan dengan teks
mentah. Jika teks kosong atau nilai hilang perlu dibedakan dari string
kosong, tangani missing value pada tahap terpisah.

## Regex untuk Kata Kunci Bencana dan Permintaan Bantuan

Regex dapat menjadi penyaring awal untuk menemukan unggahan yang
mengandung kata kunci. Contoh berikut bukan model klasifikasi dan tidak
dapat memastikan apakah unggahan benar-benar melaporkan kejadian nyata
atau meminta pertolongan.

``` python
import re

disaster_pattern = re.compile(
    r"\b(?:banjir|gempa|longsor|tsunami|kebakaran)\b",
    flags=re.IGNORECASE
)

help_pattern = re.compile(
    r"\b(?:butuh bantuan|mohon bantuan|tolong|evakuasi|butuh obat)\b",
    flags=re.IGNORECASE
)

def extract_rule_features(text: str) -> dict:
    text = text if isinstance(text, str) else ""
    return {
        "mentions_disaster_keyword": bool(disaster_pattern.search(text)),
        "mentions_help_keyword": bool(help_pattern.search(text)),
    }

samples = [
    "Mohon bantuan evakuasi untuk warga terdampak banjir.",
    "Artikel lama membahas sejarah banjir.",
    "Hari ini cuaca cerah.",
]

for sample in samples:
    print(sample, extract_rule_features(sample))
```

Output:

```text
Mohon bantuan evakuasi untuk warga terdampak banjir. {'mentions_disaster_keyword': True, 'mentions_help_keyword': True}
Artikel lama membahas sejarah banjir. {'mentions_disaster_keyword': True, 'mentions_help_keyword': False}
Hari ini cuaca cerah. {'mentions_disaster_keyword': False, 'mentions_help_keyword': False}
```

Fitur berbasis aturan dapat digunakan untuk eksplorasi, penyaringan
data, atau baseline. Namun, kata `tolong` dapat muncul dalam konteks
non-darurat, sedangkan laporan darurat mungkin tidak memakai kata kunci
yang ada. Evaluasi aturan terhadap sampel berlabel manusia dan ukur
precision, recall, serta F1-score jika aturan digunakan sebagai sistem
deteksi.

### Membedakan kata kunci dan label

Kehadiran kata `banjir` bukan bukti bahwa unggahan merupakan laporan
bencana yang valid. Demikian pula, kata `bantuan` tidak otomatis berarti
kebutuhan darurat. Pisahkan:

- **Aturan pencarian:** membantu menemukan kandidat data.
- **Label referensi:** ditentukan melalui prosedur anotasi yang jelas.
- **Prediksi model:** keluaran sistem yang harus dievaluasi.
- **Keputusan operasional:** membutuhkan validasi, konteks, dan prosedur
  manusia jika menyangkut bantuan nyata.

Jangan menggunakan Regex sebagai satu-satunya dasar untuk
memprioritaskan atau menolak permintaan bantuan darurat.

## Risiko Metodologis dalam Penelitian

### Kehilangan informasi

Menghapus emoji, hashtag, mention, tanda baca, angka, atau kata tertentu
dapat menghilangkan sinyal berguna. Bandingkan beberapa konfigurasi
preprocessing:

1.  Teks mentah dengan normalisasi whitespace saja.
2.  Teks tanpa URL.
3.  Teks dengan normalisasi tanda baca.
4.  Aturan tambahan yang didukung analisis data.

Bandingkan hasil pada pembagian data yang sama dan hindari mengubah
preprocessing berdasarkan informasi dari test set.

### Data leakage

Jangan menentukan aturan, kosakata, atau ambang keputusan dengan
memanfaatkan label test set. Jika aturan dikembangkan berdasarkan
anotasi, lakukan pengembangan pada data train atau validation, kemudian
evaluasi pada test set yang terpisah.

### Bias kata kunci

Daftar kata kunci yang terlalu sempit dapat melewatkan variasi bahasa,
singkatan, dialek, salah ketik, dan istilah lokal. Daftar terlalu luas
dapat meningkatkan false positive. Susun daftar dengan kajian literatur,
pemeriksaan data, dan masukan anotator.

### Privasi dan etika

Data media sosial dapat berisi nama, nomor telepon, lokasi, atau
informasi sensitif. Minimalkan data yang dikumpulkan, lindungi data
mentah, batasi akses, dan anonimisasi informasi pribadi bila diperlukan.
Jangan memublikasikan contoh yang dapat mengidentifikasi orang tanpa
dasar yang sesuai.

### Reproducibility

Catat versi Python, aturan Regex, flags, urutan preprocessing, tanggal
perubahan, dan contoh input-output. Simpan versi teks mentah agar hasil
pembersihan dapat ditinjau ulang.

## Pengujian Unit untuk Aturan Regex

Regex yang tampak benar pada satu contoh bisa gagal pada variasi lain.
Buat pengujian otomatis untuk perilaku yang diharapkan.

``` python
def test_clean_text():
    assert clean_text("Banjir!!!") == "Banjir!"
    assert clean_text("  Mohon   bantuan  ") == "Mohon bantuan"
    assert clean_text("Info https://contoh.id") == "Info"
    assert clean_text("") == ""

test_clean_text()
print("Semua pengujian dasar berhasil.")
```

Output:

```text
Semua pengujian dasar berhasil.
```

Tambahkan kasus yang mencerminkan data sebenarnya, termasuk string
kosong, nilai hilang, emoji, tanda baca di akhir URL, hashtag, mention,
salah ketik, dan bahasa campuran. Periksa juga kasus yang seharusnya
tidak berubah.

## Mengukur Kualitas Preprocessing

Preprocessing tidak cukup dinilai dari seberapa bersih teks terlihat.
Untuk penelitian machine learning, ukur dampaknya terhadap tujuan akhir.

Aspek yang dapat diperiksa:

- Persentase teks kosong setelah preprocessing.
- Jumlah karakter atau token sebelum dan sesudah pembersihan.
- Proporsi URL, hashtag, mention, emoji, dan angka yang terhapus.
- Contoh false positive dan false negative pada aturan penyaringan.
- Performa model pada konfigurasi preprocessing berbeda.
- Stabilitas hasil pada data dari sumber atau periode berbeda.

Jika Regex digunakan sebagai penyaring kandidat, buat sampel evaluasi
berlabel dan hitung:

- **Precision:** proporsi kandidat yang benar-benar relevan.
- **Recall:** proporsi seluruh data relevan yang berhasil ditemukan.
- **F1-score:** rata-rata harmonik precision dan recall.

Recall penting jika tujuan penyaringan adalah mengurangi risiko
melewatkan laporan relevan. Metrik dan ambang yang tepat tetap
bergantung pada tujuan penelitian dan konsekuensi kesalahan.

## Kesalahan Umum dan Cara Menghindarinya

| Kesalahan                                            | Dampak                                       | Perbaikan                           |
|:-----------------------------------------------------|:---------------------------------------------|:------------------------------------|
| Menggunakan `.` tanpa escape untuk mencari titik     | Cocok dengan karakter lain                   | Gunakan `\.`                        |
| Lupa raw string                                      | Escape Python dan Regex membingungkan        | Tulis pola sebagai `r"..."`         |
| Menggunakan `match()` untuk pencarian di tengah teks | Kecocokan tidak ditemukan jika bukan di awal | Gunakan `search()`                  |
| Menghapus seluruh tanda baca                         | Sinyal ekspresif dapat hilang                | Terapkan aturan selektif            |
| Menghapus semua karakter non-ASCII                   | Emoji dan karakter bahasa lain hilang        | Tentukan aturan berbasis tujuan     |
| Menganggap kata kunci sama dengan label              | Salah klasifikasi                            | Validasi dengan anotasi manusia     |
| Memakai Regex untuk mem-parsing HTML kompleks        | Pola rapuh                                   | Gunakan parser khusus               |
| Tidak menguji variasi data                           | Kesalahan tidak terdeteksi                   | Buat unit test dan kasus tepi       |
| Mengubah aturan setelah melihat hasil test           | Risiko data leakage                          | Kunci aturan sebelum evaluasi final |

## Latihan Praktik

Coba selesaikan tanpa melihat kunci jawaban.

### Latihan 1 — Ekstraksi angka

Dari teks `"Ada 12 rumah rusak dan 3 jembatan terdampak"`, ekstrak
seluruh rangkaian digit.

### Latihan 2 — Pencarian kata utuh

Cari kata `gempa` sebagai kata utuh dalam
`"gempa, pascagempa, dan dampak gempa"`.

### Latihan 3 — Ekstraksi hashtag

Ekstrak hashtag dari `"Pantau #InfoBencana dan #BantuanWarga"`.

### Latihan 4 — Normalisasi whitespace

Ubah `"Mohon   bantuan\nuntuk warga"` menjadi satu baris dengan satu
spasi antar kata.

### Latihan 5 — Cleaning konservatif

Buat fungsi yang menghapus URL dan merapikan whitespace, tetapi
mempertahankan emoji dan hashtag.

### Latihan 6 — Penyaring kata kunci

Buat Regex untuk mendeteksi salah satu kata `banjir`, `gempa`, atau
`longsor`, tanpa membedakan huruf besar dan kecil. Jelaskan mengapa
hasilnya belum cukup untuk menetapkan label bencana.

### Latihan 7 — Evaluasi aturan

Buat 20 contoh teks, beri label relevan/tidak relevan secara manual,
jalankan aturan Regex, lalu hitung precision dan recall. Analisis contoh
yang salah.

## Kunci Jawaban Latihan

### Jawaban 1

``` python
re.findall(r"\d+", "Ada 12 rumah rusak dan 3 jembatan terdampak")
# ['12', '3']
```

### Jawaban 2

``` python
re.findall(r"\bgempa\b", "gempa, pascagempa, dan dampak gempa", re.IGNORECASE)
# ['gempa', 'gempa']
```

### Jawaban 3

``` python
re.findall(r"(?<!\w)#[A-Za-z0-9_]+", "Pantau #InfoBencana dan #BantuanWarga")
# ['#InfoBencana', '#BantuanWarga']
```

Pola ini menggunakan karakter ASCII. Jika korpus memiliki hashtag
Unicode, sesuaikan dan uji implementasinya.

### Jawaban 4

``` python
text = "Mohon   bantuan\nuntuk warga"
clean = re.sub(r"\s+", " ", text).strip()
# 'Mohon bantuan untuk warga'
```

### Jawaban 5

``` python
def clean_social_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

### Jawaban 6

``` python
pattern = re.compile(
    r"\b(?:banjir|gempa|longsor)\b",
    re.IGNORECASE
)
```

Aturan hanya mengidentifikasi kemunculan kata tertentu. Aturan tidak
memahami apakah teks berupa berita lama, kutipan, penyangkalan, diskusi
akademik, atau laporan kejadian yang sedang berlangsung.

### Jawaban 7

Tidak ada satu nilai precision atau recall yang dapat diberikan tanpa
data dan label hasil latihan. Hitung nilai berdasarkan prediksi aturan
terhadap label referensi, lalu dokumentasikan prosedur anotasi.

## Template Eksperimen Perbandingan Preprocessing

| Eksperimen | Aturan preprocessing                      | Hipotesis                                   | Metrik evaluasi       |
|:-----------|:------------------------------------------|:--------------------------------------------|:----------------------|
| A          | Whitespace saja                           | Menjadi baseline konservatif                | F1, precision, recall |
| B          | A + penghapusan URL                       | URL mungkin tidak membantu klasifikasi teks | F1, precision, recall |
| C          | B + normalisasi tanda baca berulang       | Mengurangi variasi format                   | F1, precision, recall |
| D          | Aturan tambahan berdasarkan analisis data | Aturan tambahan memberi manfaat             | F1, precision, recall |

Gunakan pembagian train/validation/test dan konfigurasi model yang
konsisten. Jika memungkinkan, laporkan variasi hasil dari beberapa seed
atau validasi silang yang sesuai dengan struktur data. Untuk data
berbasis waktu, pertimbangkan pemisahan temporal agar evaluasi
menyerupai penggunaan pada data masa depan.

## Checklist Sebelum Regex Digunakan dalam Penelitian

- [ ] Tujuan setiap aturan dijelaskan.
- [ ] Teks mentah dipertahankan secara aman.
- [ ] Urutan preprocessing terdokumentasi.
- [ ] Pola diuji pada contoh positif dan negatif.
- [ ] Kasus tepi sudah diperiksa.
- [ ] Hashtag, emoji, angka, dan tanda baca tidak dihapus tanpa alasan.
- [ ] Data pribadi ditangani secara etis dan aman.
- [ ] Tidak ada informasi test set yang digunakan untuk menyusun aturan.
- [ ] Hasil preprocessing diperiksa manual pada sampel.
- [ ] Dampak preprocessing terhadap performa model dievaluasi.
- [ ] Kode dan versi aturan disimpan untuk reproduksibilitas.

## Referensi Dokumentasi

1.  [Python Software Foundation](https://docs.python.org/3/howto/regex.html). *Regular Expression HOWTO*.
2.  [Python Software Foundation](https://docs.python.org/3/library/re.html). *`re` — Regular expression operations*.
3.  [Regex101](https://regex101.com/). *Online regular expression tester*.

Dokumentasi resmi Python merupakan rujukan utama untuk sintaks dan
perilaku modul `re`. Periksa dokumentasi sesuai versi Python yang
digunakan dalam eksperimen.

## Penutup

Regex efektif untuk aturan tekstual yang eksplisit, ekstraksi pola, dan
sebagian tahap preprocessing. Dalam penelitian NLP berbasis media
sosial, kualitas penerapan Regex ditentukan oleh alasan metodologis,
pengujian pada data nyata, dokumentasi, serta evaluasi dampaknya
terhadap model.

Untuk penelitian terkait bencana atau permintaan bantuan darurat,
gunakan Regex sebagai komponen pendukung—misalnya ekstraksi fitur atau
penyaringan awal—bukan satu-satunya penentu urgensi. Keputusan yang
berdampak pada keselamatan manusia memerlukan validasi yang memadai,
konteks, dan pengawasan manusia.
