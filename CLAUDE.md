# Claude Code Instructions

## Role

You are modifying an existing MkDocs documentation project.

## Priority

1.  Preserve existing content.
2.  Preserve existing functionality.
3.  Improve UI/UX.
4.  Minimize code and dependencies.
5.  Keep the implementation maintainable and performant.

## Before Coding

Inspect: - `mkdocs.yml` - `docs/` - theme configuration - theme
overrides - custom CSS - custom JavaScript - plugins - dependencies -
navigation - existing homepage

Do not modify files during the initial audit.

## Rules

-   Do not rewrite documentation content unless explicitly requested.
-   Do not delete files without verifying their purpose.
-   Do not replace the entire theme without justification.
-   Prefer existing MkDocs/theme features.
-   Prefer CSS over JavaScript when possible.
-   Avoid unnecessary dependencies.
-   Avoid duplicated styles.
-   Reuse existing components and tokens.
-   Keep responsive behavior in mind for every UI change.
-   Do not make unrelated changes.

## Workflow

1.  Audit the project.
2.  Record findings in `PROJECT_AUDIT.md`.
3.  Create/update `IMPLEMENTATION_PLAN.md`.
4.  Implement one logical phase at a time.
5.  Run `mkdocs build` after major changes.
6.  Fix regressions before continuing.
7.  Review changed files.
8.  Verify desktop and mobile behavior.

## Verification

After implementation check: - build success - broken links - missing
assets - console errors - responsive layout - sidebar/navigation -
search - theme switching - code blocks - existing documentation

## Communication

Before a major change, briefly state: - files affected - purpose - risk

Keep explanations concise to reduce token usage.

## Important

Do not repeatedly rediscover project structure. Reuse findings from
`PROJECT_AUDIT.md` and `IMPLEMENTATION_PLAN.md`.

## Token Efficiency

Keep all interactions concise.

Rules:
- Do not repeat information already documented in project files.
- Read only the files required for the current task.
- Prefer targeted file inspection over scanning the entire repository repeatedly.
- Reuse `PROJECT_AUDIT.md` and `IMPLEMENTATION_PLAN.md` instead of repeating analysis.
- Do not explain obvious implementation details.
- Do not provide long summaries after each task.
- Report only changed files, verification result, and important issues.
- Do not paste large file contents into responses.
- Do not inspect unrelated files.
- Do not perform duplicate analysis.
- Work in small phases.
- Stop after completing the requested phase.
- Wait for the next instruction.

### Response Format

After each task, respond with:

Status: PASS / FAIL
Changed: <files>
Verified: <command/result>
Issues: <only if applicable>

Keep the response under 10 lines whenever possible.
