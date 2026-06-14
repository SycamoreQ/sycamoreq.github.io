---
title: another placeholder note
date: 2025-01-01
description: a brief description — what will someone learn from reading this?
tags: [compilers, ocaml]
---

This is your second placeholder note. Delete both and replace with your own writing.

## frontmatter reference

Each note needs these fields in the `---` block at the top:

| field         | type              | notes                               |
|---------------|-------------------|-------------------------------------|
| `title`       | string            | shown in listing and as the heading |
| `date`        | YYYY-MM-DD        | used for sorting                    |
| `description` | string            | one-line blurb in the listing       |
| `tags`        | array of strings  | displayed on the note page          |
| `draft`       | true / false      | set `true` to hide from listing     |

## adding notes

Drop any `.md` file into `src/content/notes/`. The listing page sorts by date descending automatically.
