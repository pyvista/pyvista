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
python3 doc/run_vale.py
```

That script holds the list of paths Vale checks, extracts the `.rst` files it needs, and
finishes by confirming the heading rule still rejects the fixture in
`tests/doc/vale/headings_invalid.rst`. Do not spell the `vale` invocation out by hand --
four copies of it had already drifted apart once.

Prose in a `.py` file is checked through the `.rst` the extraction script generates, not
from the `.py` itself. The extracted file has the same line numbers as the source
(`doc/extract_rst_from_py_for_vale.py:31:1`) -- read an alert's `.vale/examples/...` or
`.vale/pyvista/...` path, swap it back to `examples/...` or `pyvista/...` with a `.py`
extension, and that is the line to fix.

Vale does still open `.py` files directly, for the rules `doc/.vale.ini`'s `[*.py]`
section turns on -- today just `Google.Exclamation`. **An alert reported against a
`pyvista/...py` path rather than a `.vale/...rst` one came from that scan**, which sees
the raw file: numpydoc signature lines, `See Also` entries and doctest blocks all reach
it as ordinary prose, with none of the handling described below. Enabling a prose rule
there means signing up for that. `PyVista.Repetition` was enabled there once and needed
an exceptions list of two dozen type names to stay quiet; moving it to the extracted
`.rst` retired the whole list.

**Trust a live Vale run over a mental model of what it checks.** This session got this
wrong twice: once assuming a rule (`Google.Headings`'s exceptions list) was needed by
guessing rather than removing an entry and re-running, and once assuming `See Also` had
to stay skipped without first trying the alternative and reading what actually surfaced.
Toggle the specific `.yml` file or the extraction script, regenerate, run Vale, and read
the real diff in alert count before deciding a rule change is safe.

## What the extractor does and does not see

- Every numpydoc section shaped `name : type` over an indented description has its
  signature line blanked before Vale runs; the description is checked like any other
  prose. That is `Parameters`, `Other Parameters`, `Attributes`, `Methods`, `Returns`,
  `Yields`, `Raises`, `Warns` and `Receives` -- the `STRUCTURED_SECTIONS` set in
  `doc/extract_rst_from_py_for_vale.py`, which is the list to read rather than this one.
  Blanking the signature line is what stops a description restating the type it
  documents (`pyvista.PolyData` over "PolyData mesh.") from reading as a doubled word.
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
- **An accepted word is exempt from every rule, not just `Vale.Spelling`.** Vale drops
  vocabulary terms before any check runs, so a term in `accept.txt` can never be reported
  by `PyVista.Repetition`, `Google.Latin`, or anything else. `PolyData` sat in that rule's
  exceptions list for exactly this reason: it was already unreachable, and the entry did
  nothing. If a rule mysteriously will not fire on a word you expect it to, grep
  `accept.txt` before assuming the rule is broken.
- **The file is dual-purpose.** `pyproject.toml`'s `codespell` config also uses it as
  `ignore-words`. A word can be dead to Vale (no prose ever uses it, because it only shows
  up in a raw comment, a test fixture, or a variable name) and still required by
  codespell, which scans a wider set of paths (`tests`, `examples_trame`, ...) and file
  content types. Before deleting an apparently-unused entry, run
  `pre-commit run codespell --all-files` after the deletion, not just Vale.

## Common per-alert judgment calls

- **Project name vs. code, for any dual-cased pair** (`PyVista`/`pyvista`, `VTK`/`vtk`,
  `NumPy`/`numpy`, `Matplotlib`/`matplotlib`, `SciPy`/`scipy`, ...). Use the capitalized
  form whenever the sentence is naming the project or library ("built directly from
  NumPy arrays", "wraps most of VTK"). Use the lowercase form only where it is literally
  code: an import, a module path (`numpy.ndarray`), a parameter default, a snake_case
  identifier. This is not a `Vale.Spelling` fix (both spellings are real words) and it is
  not machine-checked -- `Vale.Terms` would be the obvious rule for it, but it is
  deliberately disabled (see the comment above it in `doc/.vale.ini`): it votes on whichever
  casing is more frequent in a file or corpus and flags the other one, so it flagged a
  module's own correct `"""NumPy plotting module."""`-style summary line as wrong the one
  time it was tried, purely because lowercase `numpy` (real code) was more common
  elsewhere in that file. Read for this by hand, especially in a docstring's `Examples`
  section, which is exactly the kind of narrative prose most likely to get it wrong and,
  until recently, was invisible to Vale entirely.
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
- **A letter suffixed directly onto a backtick-wrapped term breaks RST silently.**
  Writing the plural of `int` as `int``s` (no separator between the closing backticks
  and the `s`) passes Vale and even `ruff`, but fails a real Sphinx build with "Inline
  literal start-string without end-string": RST's inline-markup rules require
  whitespace or punctuation immediately after a closing pair of backticks, not a bare
  letter. Fix it with an escaped space between the backticks and the suffix, and make
  the docstring a raw string (`r"""`) so the backslash survives instead of being
  Python's own invalid-escape-sequence warning. Vale cannot see this class of bug at
  all -- it only ever checks the generated `.rst` shadow copy, never runs a real Sphinx
  build. Confirm a suspicious backtick construct with `docutils.core.publish_doctree`
  on the actual docstring text, not by trusting Vale's silence.
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
