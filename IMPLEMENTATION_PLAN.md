# Implementation Plan

Based on `PROJECT_AUDIT.md`. Implement one phase at a time; verify before the next; do not commit unless explicitly requested. Never edit `docs/english/**` content or filenames.

## Phase 0 — Audit ✅

-   Done: built-in `mkdocs` theme, one table-only CSS file, no JS/plugins/overrides, Material 9.7.7 already installed, `requirements.txt` deleted in working tree.

## Phase 1 — Material Migration Validation

Goal: prove the switch to Material is safe **before** any redesign. Work in a throwaway config (e.g. `mkdocs.validate.yml` in the scratchpad, or a branch); do not touch `docs/`.

1.  **Baseline:** build the current site to a temp dir; record page count, output URLs, and `git ls-files docs | xargs sha256sum`.
2.  **Candidate config:** `theme: material` + existing `markdown_extensions` (`tables`, `admonition`, `pymdownx.tilde`) + existing `extra_css`. No new features yet.
3.  **Checks:**

    | # | Check | Method |
    |---|-------|--------|
    | 1 | Build succeeds | `mkdocs build` completes |
    | 2 | Page count | Same pages as baseline |
    | 3 | URLs unchanged | Output paths identical to baseline |
    | 4 | Content preserved | `git diff --stat docs/` empty; checksums match |
    | 5 | Tables readable | Visual check of lessons at 320/768/1440px, with and without old `extra.css` |
    | 6 | Search works | Serve site; query a few lesson terms |
    | 7 | Admonitions work | Lesson-1 `!!!` block renders |
    | 8 | `pymdownx.tilde` works | `~~text~~` renders as `<del>` in all 3 lessons |
    | 9 | Strict build clean | `mkdocs build --strict` with no warnings/errors |

4.  **Decision gate:** all 9 pass → proceed. Any failure → fix in config/CSS, or document and reconsider before Phase 2.
5.  **Deps decision:** ask whether to recreate `requirements.txt` (proposed minimal: `mkdocs`, `mkdocs-material`, `pymdown-extensions`, pinned). Do not restore the 158-package freeze.

**Done when:** all checks pass and results are recorded in `PROJECT_AUDIT.md`.

## Phase 2 — Design System

-   Apply the validated Material config to `mkdocs.yml` (palette light/dark toggle, fonts, `language`).
-   Rewrite `extra.css` as tokens only (colors, spacing, radius, borders) using Material CSS variables; remove `!important`/`.md-typeset`-redundant selectors.
-   **Verify:** one centralized token block; light/dark both work; build clean.

## Phase 3 — Global Layout

-   `mkdocs.yml` features, start with only: `navigation.sections`, `navigation.path` (breadcrumbs), `navigation.footer`, `navigation.top`, `toc.follow`.
-   `navigation.instant` and `navigation.tabs` are optional; enable only if justified after testing.
-   Restore/define `nav` (keep lesson URLs; fix display titles only).
-   Footer + repo link via config.
-   **Verify:** sidebar, TOC, header, mobile drawer.

## Phase 4 — Documentation Components

-   Add only needed extensions: `pymdownx.highlight`, `pymdownx.superfences`, `content.code.copy` feature.
-   CSS for tables (scrollable, theme-aware, zebra optional), blockquotes, admonitions, images, links.
-   **Verify:** all 3 lessons render as in Phase 1; tilde, admonition, tables intact.

## Phase 5 — Homepage

-   Restructure `docs/index.md` (keep existing Indonesian copy): hero, CTA to Beginner lessons, learning path/popular lessons as grid cards, GitHub link. Needs `attr_list` + `md_in_html` only if used.
-   Prefer Markdown + CSS; override template only if unavoidable.
-   **Verify:** purpose clear above the fold on desktop and mobile.

## Phase 6 — Responsive & Accessibility

-   Test 320 / 768 / 1024 / 1440px: no horizontal page overflow (tables scroll internally).
-   Keyboard navigation, visible focus, touch targets, contrast (AA) in both modes, alt text where images exist.

## Phase 7 — Performance & Cleanup

-   Remove unused CSS and any features not needed; no JS added.
-   Confirm dependency list is minimal and pinned; update `README.md` only where inaccurate.
-   Consider self-hosted/limited fonts (privacy/performance) if external fonts are used.

## Phase 8 — QA

-   `mkdocs build --strict` clean; broken links, missing assets, console errors.
-   Navigation, search, theme switching, code copy, desktop + mobile.
-   Confirm `git diff` shows `docs/english/**` unchanged and URLs identical to baseline.
-   Check PRD acceptance criteria.

## Git Guidance

Small commits, e.g. `chore: audit mkdocs project`, `chore: validate material migration`, `feat: add design system`, `feat: redesign layout`, `feat: improve doc components`, `feat: redesign homepage`, `fix: responsive and accessibility`, `chore: cleanup and deps`.
