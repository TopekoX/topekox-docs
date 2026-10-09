# Project Audit

Audited 2026-10-07. No project files modified (only this file written).

## 1. Project

-   MkDocs version: 1.6.1 (`.venv`)
-   Theme: built-in `mkdocs` (NOT Material), despite README claiming Material
-   Python environment: `.venv/` (gitignored); Material 9.7.7, pymdown-extensions 12.1 installed but unused
-   Build command: `.venv/bin/mkdocs build` — succeeds, no warnings

## 2. Structure

-   `mkdocs.yml`: 17 lines. `site_name`, `theme: mkdocs`, extensions `tables`, `pymdownx.tilde`, `admonition`, `extra_css`. `nav` is commented out (auto-nav from file tree).
-   `docs/`: `index.md` (Indonesian intro), `english/beginner/leasson-{1,2,3}.md` (1028/1203/1297 lines), `stylesheets/extra.css`.
-   Theme overrides: none (no `custom_dir`).
-   CSS: `docs/stylesheets/extra.css` (~20 lines, tables only).
-   JavaScript: none.
-   Plugins: none declared (default `search` only).
-   Dependencies: `requirements.txt` is tracked in HEAD (158 pinned packages, incl. jupyter, streamlit, scipy, weasyprint, ollama — a full `pip freeze`, only mkdocs/material/pymdown relevant) but is **deleted in the working tree** (uncommitted `D`).

## 3. Current UI

### Header
Default mkdocs theme: Bootstrap navbar, site name + search + prev/next icons. Dated look.

### Sidebar
Auto-generated from file tree; folders/files titled from filenames/H1 (`leasson-1`). Bootstrap side nav; collapses to hamburger on mobile.

### Documentation Layout
Bootstrap grid: nav column + content + TOC column. No breadcrumbs. Prev/Next exist only in navbar. Content width uncapped by design tokens.

### Homepage
Plain `index.md` rendered as a normal page (H1 "Home", paragraphs, list, quote). No hero, CTA, or feature/popular-lesson links.

### Footer
Default mkdocs footer (prev/next links only); no project/GitHub links.

### Code Blocks
Only 2 fenced blocks across all lessons (lesson-2). No `pymdownx.highlight`/`superfences`; no copy button. Low impact currently, but needed for PRD.

### Search
Built-in lunr search (default plugin). Works; no Indonesian/English tuning.

### Dark/Light Mode
Not supported (mkdocs theme has no palette toggle).

### Mobile
Bootstrap collapse nav works, but tables forced `width:max-content` + `white-space:nowrap` (see below) likely overflow on narrow screens.

## 4. Existing Customization

-   `extra.css` table rules: borders `#777`, header bg `#eaeaea`, padding, `width:max-content; max-width:100%`, `white-space:nowrap`, heavy `!important` and redundant selectors (`article table, .md-typeset table, table` — `.md-typeset` is a Material class, so these were likely written for Material). Purpose: compact tables. Hard-coded light colors → will break dark mode.
-   Extensions: `admonition` (used once, lesson-1), `pymdownx.tilde` (used for `~~strike~~` in blockquotes, all 3 lessons), `tables` (heavy: 125/132/158 table rows).

## 5. Problems

-   P0: README says Material but config uses `mkdocs` theme; `requirements.txt` deleted in working tree and bloated (do not restore as-is).
-   P1: Dated default theme; no dark mode; no homepage design; no breadcrumbs/footer; unstyled nav titles ("leasson-1" typo in filenames, title comes from H1).
-   P1: Table CSS non-responsive (`nowrap` + `max-content` → overflow), `!important` everywhere, hard-coded colors.
-   P1: Very long pages (1000–1300 lines, 23–36 H2s) rely on TOC quality.
-   P2: No code copy/highlight config; no nav ordering/labels (nav commented out); no favicon/footer/repo links; `index.md` H1 "Home" is generic.

## 6. Constraints

-   Keep all `docs/**/*.md` content untouched, including filenames/URLs (`leasson-N` typo is part of URLs — do not rename).
-   Keep extensions `tables`, `admonition`, `pymdownx.tilde`.
-   Tables are core content; they must remain readable (scrollable, not clipped).
-   Search must keep working. Build must stay warning-free.
-   Commented `nav` block in `mkdocs.yml` documents intended structure.

## 7. Recommended Approach

Switch to Material for MkDocs (already installed, already documented in README, already intended by `.md-typeset` selectors) — justified because the built-in theme lacks dark mode, instant nav, TOC/breadcrumb/copy features without custom JS. Configure via `mkdocs.yml` features + palette toggle + `pymdownx.highlight/superfences/tabbed` as needed; replace `extra.css` with a small token-based stylesheet (tables scroll wrapper via CSS, `overflow-x:auto`, theme variables instead of hard-coded colors); homepage via Markdown + Material grid cards (no override templates, no JS). Add a minimal `requirements.txt` (mkdocs, mkdocs-material, pymdown-extensions).

## 8. Files to Change

| File | Reason | Risk |
|------|--------|------|
| `mkdocs.yml` | Material theme, palette, features, extensions, nav titles | Medium: theme switch changes all rendering |
| `docs/stylesheets/extra.css` | Rewrite with tokens, responsive tables | Medium: tables are dominant content |
| `docs/index.md` | Hero/CTA/learning path (content preserved, restructured) | Low–Med: Indonesian copy must be kept |
| `requirements.txt` | Replace with minimal deps | Low |
| `README.md` | Only if needed to stay accurate | Low |
| `docs/overrides/` (new, only if needed) | Footer/extras | Low |

## 9. Files to Preserve

| File/Area | Reason |
|-----------|--------|
| `docs/english/beginner/*.md` | Core content; URLs depend on filenames |
| `.gitignore`, `.venv/` | Environment/config |
| `CLAUDE.md`, `PRD.md`, `IMPLEMENTATION_PLAN.md` | Project instructions |

## 10. Audit Status

-   [x] Repository inspected
-   [x] Theme identified
-   [x] Configuration reviewed
-   [x] Custom CSS reviewed
-   [x] Custom JS reviewed (none)
-   [x] Plugins reviewed (none)
-   [x] Navigation reviewed
-   [x] Homepage reviewed
-   [x] Risks documented

## Risks

-   Theme switch alters rendering of ~3.5k lines of table-heavy Markdown; visually verify each lesson.
-   Material feature set (e.g. `attr_list`, grid cards) requires extra Markdown extensions; homepage must not change lesson content.
-   Uncommitted `requirements.txt` deletion: decide intent before recreating.
-   Material 9.x is in maintenance mode upstream; pin the version.
-   Table `nowrap` removal may reflow wide tables; use horizontal scroll to compensate.

## Phase 1 Validation Result (Material Migration)

Run 2026-10-07 with a scratchpad config (`theme: material` + existing extensions + existing `extra.css`, `docs_dir` → project `docs/`). No project file was modified. Material 9.7.7 / MkDocs 1.6.1.

| # | Check | Result |
|---|-------|--------|
| 1 | Build succeeds | PASS |
| 2 | Page count | PASS — 4 pages (index + 3 lessons), same as baseline |
| 3 | URLs unchanged | PASS — output paths identical to baseline |
| 4 | Content preserved | PASS — `docs/` checksums identical before/after; no `docs/` change in `git status` |
| 5 | Tables readable | PASS — lesson-3 table viewed at 500px (Chrome headless minimum width); measured 500/768/1440px: no page-level horizontal overflow, with and without old `extra.css`. Not tested at 320px. |
| 6 | Search works | PARTIAL — Material search index builds (278 entries vs 282 baseline; the 4 missing are per-page H1 entries that Material merges). Interactive search UI not exercised. |
| 7 | Admonitions work | PASS — lesson-1 renders 2 `admonition` blocks, same as baseline |
| 8 | `pymdownx.tilde` works | PASS — 1 `<del>` per lesson, same as baseline |
| 9 | Strict build clean | PASS — `mkdocs build --strict` exits 0. Material prints an informational "MkDocs 2.0" notice banner on every build; it is not a MkDocs warning. |

Observations:

-   Table counts differ by 1 per lesson (baseline 15/14/24 vs 14/13/23) because the built-in theme adds its own keyboard-shortcuts table; lesson content is unchanged.
-   Old `extra.css` works under Material (its `.md-typeset` selectors now apply): heavy grid borders, grey header. Without it Material's default tables also read well. Phase 2 replaces it.
-   The MkDocs 2.0 notice is a future-compat risk, not a blocker: pin `mkdocs<2`.

**Decision gate: PASSED.** Gaps to recheck in Phase 6/8: interactive search and 320px width.
Open: `requirements.txt` is still deleted in the working tree; decide whether to recreate a minimal pinned file (`mkdocs<2`, `mkdocs-material`, `pymdown-extensions`).

## Phase 8 Final QA Result

Run 2026-10-08 against a fresh strict build served locally, driven by headless Chrome 155 (DevTools protocol, scratchpad scripts, no project changes). Viewports 320/768/1024/1440px, each in light and dark (emulated `prefers-color-scheme`, plus a toggle-click test).

| Check | Result |
|---|---|
| `mkdocs build --strict` | PASS — exit 0, no warnings (only Material's informational MkDocs 2.0 banner) |
| Pages / URLs | PASS — 4 pages, output paths identical to Phase 1 baseline |
| `docs/english/**` unchanged | PASS — checksums match baseline, `git status` clean for that path |
| Page-level horizontal overflow | PASS — 4 pages × 4 widths × 2 schemes, `scrollWidth <= viewport` |
| Tables | PASS — 50 tables; at 320px 17 need scrolling and all are reachable via Material's scroll wrapper; none overflow at ≥768px |
| Contrast (≥4.5:1) | PASS — h1, body, hero, cards, blockquote, table th/td, admonition, code, nav, secondary button, links, both schemes |
| Light/dark | PASS — emulation gives `default`/`slate`; toggle cycles schemes and changes background |
| Search | PASS — query returns results at 1440px (light, dark) and 320px |
| Code copy | PASS — clipboard equals code, "Copied to clipboard" shown |
| Admonitions | PASS — render with correct fg/bg in both schemes |
| Mobile nav | PASS — drawer opens at 320px, lists nav, navigates |
| Breadcrumbs, TOC, prev/next | PASS — Home › English › Beginner; TOC present and clickable; prev/next hrefs and navigation correct |
| Homepage | PASS — hero, 2 CTAs, cards; links resolve |
| Missing assets | PASS — all internal links/assets return 200 |
| Console errors | 1 benign: GitHub API `releases/latest` → 404 (see below) |

Remaining issue: with `repo_url` set, Material fetches repo facts from `api.github.com/.../releases/latest`. The repo is public but has no releases, so the browser logs a 404 (the repo-info fallback succeeds). Not fixable via config without dropping `repo_url`; disappears once a release/tag exists.

Notes: Material persists the resolved palette in `localStorage`; automated schema tests must clear it between schemes. No fixes were required.

## Final Cleanup (post-QA)

-   Removed `repo_url` from `mkdocs.yml` (header repo widget caused the GitHub `releases/latest` 404). Homepage "GitHub" button still links directly to `https://github.com/TopekoX/topekox-docs`.
-   Re-verified: `mkdocs build --strict` clean; URLs unchanged; 4 pages × 2 widths (320/1440) × 2 schemes loaded with zero console/network errors and no `api.github.com` request.
-   README configuration/structure examples updated to match the current project; `requirements.txt` kept at 3 pinned packages.

## MathJax Support (2026-10-09)

-   Added `pymdownx.arithmatex` (`generic: true`) + `docs/javascripts/mathjax.js` + MathJax 3 CDN (`unpkg`) via `extra_javascript`. No new Python dependency (`pymdown-extensions` already installed).
-   Verified via temporary `docs/test-math.md` (built, HTML-inspected, then removed): inline/display/fraction/sum/aligned multi-line/matrix equations all emit `<span class="arithmatex">`/`<div class="arithmatex">`; code blocks, tables, and admonitions on the same page render unaffected.
-   `mkdocs build --strict` clean; `docs/english/**` untouched.
