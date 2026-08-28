# topekox-docs

📚 Dokumentasi pribadi untuk project coding - dibangun dengan MkDocs

## 📖 Deskripsi

**topekox-docs** adalah repository dokumentasi komprehensif untuk mencatat, mengorganisir, dan mengelola dokumentasi berbagai project coding. Menggunakan MkDocs sebagai static site generator untuk menghasilkan dokumentasi yang clean, responsive, dan mudah dicari.

Dokumentasi ini berfungsi sebagai knowledge base pribadi yang dapat diakses kapan saja untuk referensi cepat tentang:
- Setup dan konfigurasi project
- Catatan teknis dan best practices
- Snippets dan contoh code
- Troubleshooting dan solutions
- Architecture decisions dan learnings

## ✨ Fitur

- 📱 **Responsive Design** - Dapat diakses dari desktop, tablet, dan mobile
- 🔍 **Search Functionality** - Pencarian cepat across semua dokumentasi
- 🎨 **Material Theme** - UI yang modern dan user-friendly dengan Material Design
- 📂 **Organized Structure** - Struktur folder yang jelas dan hirarkis
- ⚡ **Fast Build** - Static site yang cepat dan efficient
- 🚀 **Easy Deployment** - Mudah di-deploy ke berbagai platform
- 🔗 **Cross-linking** - Link internal antar dokumentasi
- 📝 **Markdown-based** - Mudah ditulis dan di-maintain dengan Markdown

## 🚀 Quickstart

### Prerequisites

- Python 3.6 atau lebih tinggi
- pip (Python package manager)

### Instalasi

1. Clone repository ini:
```bash
git clone https://github.com/topekox/topekox-docs.git
cd topekox-docs
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Atau install MkDocs dan theme secara manual:
```bash
pip install mkdocs mkdocs-material
```

### Menjalankan Documentation Locally

Untuk preview dokumentasi di browser dengan auto-reload:

```bash
mkdocs serve
```

Buka browser dan kunjungi `http://localhost:8000`

### Build untuk Production

Untuk generate static HTML files:

```bash
mkdocs build
```

Output akan di-generate di folder `site/`

## 📁 Struktur Directory

```
topekox-docs/
├── docs/                          # Folder utama dokumentasi
│   ├── index.md                   # Homepage dokumentasi
│   ├── getting-started/           # Panduan untuk memulai
│   │   └── setup.md
│   ├── projects/                  # Dokumentasi per project
│   │   ├── project-a.md
│   │   └── project-b.md
│   ├── coding/                    # Tips & snippets coding
│   │   ├── python.md
│   │   ├── javascript.md
│   │   └── database.md
│   ├── tools-setup/               # Setup tools & environment
│   │   ├── development-env.md
│   │   ├── docker.md
│   │   └── git-workflow.md
│   ├── troubleshooting/           # Solutions untuk common issues
│   │   └── common-errors.md
│   ├── resources/                 # Referensi eksternal & links
│   │   └── links.md
│   └── assets/                    # Images, diagrams, etc
│       └── images/
├── mkdocs.yml                     # Konfigurasi MkDocs
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md                      # File ini
```

## ⚙️ Konfigurasi

File `mkdocs.yml` mengontrol konfigurasi site. Berikut template dasar:

```yaml
site_name: TopekoxDocs
site_description: Dokumentasi pribadi untuk project coding
site_author: Your Name

theme:
  name: material
  language: id
  features:
    - navigation.instant
    - navigation.tracking
    - navigation.tabs
    - search.suggest
    - search.highlight
    - toc.follow
    - content.code.copy

plugins:
  - search
  - minify

nav:
  - Home: index.md
  - Getting Started: getting-started/setup.md
  - Projects:
      - Project A: projects/project-a.md
      - Project B: projects/project-b.md
  - Coding:
      - Python: coding/python.md
      - JavaScript: coding/javascript.md
      - Database: coding/database.md
  - Tools Setup:
      - Development Environment: tools-setup/development-env.md
      - Docker: tools-setup/docker.md
      - Git Workflow: tools-setup/git-workflow.md
  - Troubleshooting: troubleshooting/common-errors.md
  - Resources: resources/links.md
```

## 📝 Cara Menggunakan

### Menambah Dokumentasi Baru

1. Buat file `.md` baru di folder yang sesuai dalam folder `docs/`
2. Tulis dokumentasi dengan format Markdown
3. Update `mkdocs.yml` untuk menambahkan entry di bagian `nav`
4. Save dan jalankan `mkdocs serve` untuk preview

### Format Markdown

Berikut contoh struktur dokumentasi:

```markdown
# Judul Halaman

## Section 1
Penjelasan singkat tentang section ini.

### Subsection 1.1
Detail lebih lanjut.

### Subsection 1.2
Lebih banyak detail.

## Section 2
Konten lainnya.

### Code Example
\`\`\`python
# Python code example
def hello_world():
    print("Hello, World!")
\`\`\`

### Tips & Notes
!!! note "Catatan"
    Ini adalah catatan penting

!!! warning "Peringatan"
    Ini adalah warning

!!! tip "Tips"
    Ini adalah tips berguna

!!! danger "Terjadi Kesalahan"
    Proses instalasi gagal! Periksa kembali koneksi Anda.
```

## 🔧 Customization

### Mengubah Theme Color

Edit `mkdocs.yml`:
```yaml
theme:
  palette:
    primary: blue
    accent: cyan
```

### Menambah Extensions

Update `mkdocs.yml` untuk menambah markdown extensions:
```yaml
markdown_extensions:
  - pymdownx.arithmatex
  - pymdownx.betterem
  - pymdownx.caret
  - pymdownx.details
  - pymdownx.highlight
  - pymdownx.superfences
  - pymdownx.tasklist
  - tables
  - toc
```

## 🚀 Deployment

### Deploy ke GitHub Pages

1. Pastikan repository Anda di GitHub
2. Build dokumentasi:
   ```bash
   mkdocs build
   ```
3. Deploy ke GitHub Pages:
   ```bash
   mkdocs gh-deploy
   ```

### Deploy ke Platform Lain

**Netlify:**
- Connect GitHub repo ke Netlify
- Set build command: `mkdocs build`
- Set publish directory: `site`

**Vercel:**
- Import project ke Vercel
- Set build command: `pip install -r requirements.txt && mkdocs build`
- Set output directory: `site`

## 📚 Tips Penggunaan

1. **Keep it Updated** - Update dokumentasi seiring dengan development
2. **Use Categories** - Organisir dengan folder yang jelas
3. **Link Internally** - Gunakan link internal untuk cross-reference: `[Link Text](../path/to/file.md)`
4. **Include Examples** - Selalu sertakan code examples untuk dokumentasi teknis
5. **Write for Future Self** - Tulis seperti kamu sedang mengajar orang lain
6. **Use Admonitions** - Gunakan `!!! note`, `!!! warning`, dll untuk highlight penting
7. **Index Regularly** - Restructure dokumentasi saat sudah besar

## 🔗 Useful Links

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material Theme](https://squidfunk.github.io/mkdocs-material/)
- [Markdown Guide](https://www.markdownguide.org/)
- [Python Markdown Extensions](https://python-markdown.github.io/)

## 📋 Checklist untuk Dokumentasi Baru

Sebelum commit dokumentasi baru, pastikan:

- [ ] File markdown ditulis dengan jelas dan terstruktur
- [ ] Code examples tested dan berfungsi
- [ ] Links (internal & external) bekerja dengan baik
- [ ] Tidak ada typo atau grammar errors
- [ ] File sudah ditambahkan ke `mkdocs.yml`
- [ ] Dokumentasi sudah di-preview dengan `mkdocs serve`
- [ ] File tidak terlalu panjang (pertimbangkan split jika > 2000 words)

## 🤝 Contributing

Untuk dokumentasi pribadi ini, tips untuk maintain quality:

1. Review dokumentasi lama secara berkala
2. Update informasi yang sudah outdated
3. Reorganisir ketika struktur menjadi tidak efisien
4. Backup dokumentasi secara regular

## 📄 License

Personal documentation - feel free to adapt untuk kebutuhan pribadi Anda

## 📞 Notes

Repository ini adalah dokumentasi pribadi untuk project coding. Gunakan sebagai template untuk dokumentasi Anda sendiri dan customize sesuai dengan kebutuhan.

---

**Happy Documenting! 📚✨**