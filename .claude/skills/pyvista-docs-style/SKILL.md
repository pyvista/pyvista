---
name: pyvista-docs-style
description: Fix or extend the Vale documentation-style setup, or edit prose in a docstring, example, or .rst file. Load before touching doc/.vale.ini, doc/styles/, doc/extract_rst_from_py_for_vale.py, or accept.txt, and before any prose-only edit pass across many files.
---

# PyVista documentation style

`CONTRIBUTING.rst`'s `Documentation Style` section is normative. Read it first; this
skill is the operational and agent-facing half, not a second copy of the rules.

## Before editing anything

Run Vale locally the same way CI does, from the repository root:

```bash
python3 doc/extract_rst_from_py_for_vale.py examples .vale/examples
python3 doc/extract_rst_from_py_for_vale.py pyvista .vale/pyvista --mode docstrings
vale --config doc/.vale.ini doc pyvista examples CONTRIBUTING.rst .vale/examples .vale/pyvista
```

Vale never sees a `.py` file directly. It sees the `.rst` files the extraction script
generates, with the same line numbers as the source (`doc/extract_rst_from_py_for_vale.py:31:1`)
-- read an alert's `.vale/examples/...` or `.vale/pyvista/...` path, swap it back to
`examples/...` or `pyvista/...` with a `.py` extension, and that is the line to fix.

**Trust a live Vale run over a mental model of what it checks.** This session got this
wrong twice: once assuming a rule (`Google.Headings`'s exceptions list) was needed by
guessing rather than removing an entry and re-running, and once assuming `See Also` had
to stay skipped without first trying the alternative and reading what actually surfaced.
Toggle the specific `.yml` file or the extraction script, regenerate, run Vale, and read
the real diff in alert count before deciding a rule change is safe.

## What the extractor does and does not see

- `Parameters`' `name : type` signature line is blanked before Vale runs; its description
  is checked like any other prose.
- `Returns`, `Attributes`, `Raises`, `Yields`, `Warns`, `Methods`, `Other Parameters`,
  and `Receives` are **not** blanked -- their content is checked as ordinary prose. If a
  change here starts producing type-name/class-name false positives, that is a sign the
  described value needs backticks, not that the section needs skipping again.
- `See Also` **is** fully skipped, deliberately. It is numpydoc's own bare-name
  cross-reference DSL (auto-linked at doc-build time), not prose -- backtick-wrapping a
  `See Also` entry fights that convention instead of fixing anything. Don't add prose
  rules to this section's content and don't stop skipping it without checking numpydoc's
  actual rendering first.
- Plain Python comments (`#`, outside a gallery `# %%` cell) are out of scope by design.
  They are not user-facing documentation; don't sweep them into a style pass.
- A `.py` file living under `doc/source/` (`make_tables.py`, `conf.py`, ...) is not
  extracted at all -- only `pyvista/` and `examples/` are. Data literals that are not real
  docstrings (`ast.get_docstring` only walks `Module`/`ClassDef`/`FunctionDef`/
  `AsyncFunctionDef`) are invisible too: a module-level `_HELP_TEXT = """..."""` constant,
  or a `dict` value used as documentation, needs a manual look if you are chasing a
  pattern Vale should have caught but didn't.

## Vocabulary discipline (`accept.txt`)

`CONTRIBUTING.rst` has the ordered checklist (reword, hyphenate/split, backtick, then
accept). The two failure modes worth naming directly:

- **Check history before adding _or_ removing a word.** `git log -S<word> --oneline --all
-- doc/styles/config/vocabularies/pyvista/accept.txt` (the file was
  `doc/styles/Vocab/pyvista/accept.txt` before the Vale 3 migration -- check both paths
  historically). This session re-added `unwarped` as an accepted word without noticing an
  earlier commit had deliberately removed it and hyphenated the one real usage instead.
  The fix for a Vale.Spelling hit you have not seen before might already have a decided
  answer sitting in the log.
- **The file is dual-purpose.** `pyproject.toml`'s `codespell` config also uses it as
  `ignore-words`. A word can be dead to Vale (no prose ever uses it, because it only shows
  up in a raw comment, a test fixture, or a variable name) and still required by
  codespell, which scans a wider set of paths (`tests`, `examples_trame`, ...) and file
  content types. Before deleting an apparently-unused entry, run
  `pre-commit run codespell --all-files` after the deletion, not just Vale.

## Common per-alert judgment calls

- **Oxford comma false positives.** `Google.OxfordComma`'s regex (`(?:[^,]+,){1,}\s\w+\s
(?:and|or)`) fires on any comma followed later by "word and/or", not only on real
  three-item lists -- an introductory clause plus a two-item verb list ("First, create and
  plot the grid") or a fixed phrase ("whether or not") both trigger it wrongly. Reword the
  sentence to remove the false trigger; do not insert a comma that would be grammatically
  wrong just to silence the rule.
- **`Google.Latin`** swaps only match a trailing space or comma (`e.g. ` / `e.g.,`) --
  `e.g.:` (colon) is invisible to it. Grep for the colon form separately if you are
  chasing every instance by hand instead of relying on the rule firing.
- **Self-referencing docstrings.** A property or parameter's docstring should never just
  wrap its own name in backticks (`"""Return or set the `tube_width`."""`) --
  describe it in plain English instead. Backticks are for naming a genuinely different
  identifier (a sibling property, an external package, a VTK class); if the docstring is
  about `tube_width` itself, say "the tube width."
- **String-literal values vs. bare identifiers.** A parameter's accepted string value
  (`mode='cell_tree'`) is rendered ` ``'cell_tree'`` ` -- quotes inside the
  backticks -- not just ` `cell_tree` `, which reads as an identifier rather than a
  string.
- **Heading exceptions are per-word, not per-heading.** Before adding a word to
  `doc/styles/Google/Headings.yml`'s `exceptions:` list, confirm with a live Vale run that
  more than one heading actually needs it. If only one heading in the whole corpus uses
  the word, fixing that heading's wording is better than a permanent exception for it.
- **A bulk text substitution can bleed past the line you meant.** A script that replaces
  an exact heading string can also match the same text if it appears again as an ordinary
  sentence right below the heading -- check every match site, not just the first.

## After any change to error-message or help text

Docstring rewording is invisible to tests. An exception message or CLI `--help` string is
not: `pytest.raises(..., match=...)` and CLI-help assertions often quote the exact string
you are about to reword (an Oxford comma insertion, a `word(s)` -> `words` fix, a Latin
swap). Before committing a prose fix inside an `f"..."` error message, a `msg = "..."`
assignment, or a `Parameter(help="...")` string, grep `tests/` for a distinctive
substring of the _old_ text and update the matching assertion in the same commit -- do
not find out from CI.
