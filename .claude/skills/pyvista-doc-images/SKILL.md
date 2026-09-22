---
name: pyvista-doc-images
description: Repair the documentation image cache after the "Test Documentation" job fails. Load whenever a docs CI run reports image errors, a gallery example gains or loses a plot, or a rendering docstring example is edited — the fix is mostly renames, and blanket-copying the failed images gets it wrong.
---

# Documentation image cache

`tests/doc/doc_image_cache/` holds a JPG for every image the documentation renders. The
`Test Documentation` job in `docs.yml` rebuilds them and diffs each one against its cached
slot. `pyvista-testing` covers the plotting cache and the fixture that drives it; this
skill is only the documentation cache and only the repair.

Most failures here are not regressions. They are the same picture arriving under a
different filename, and the repair is a `git mv`. Replacing those files instead works —
CI goes green either way — but it rewrites binaries that never changed, which buries the
one or two images a reviewer actually needs to look at.

## Two filename families

| Pattern                                         | Source                         | What moves it                                                         |
| ----------------------------------------------- | ------------------------------ | --------------------------------------------------------------------- |
| `sphx_glr_<example>_NNN.jpg`                    | gallery example in `examples/` | a **sequential index**; adding a plot shifts every later image up one |
| `pyvista-<Class>[-<member>]-<16 hex>_NN_NN.jpg` | rendering docstring example    | a **content hash** of the example source                              |

Each has a `_vtksz` twin for the interactive scene, and gallery examples add one
`_thumb.jpg` chosen by `sphinx_gallery_thumbnail_number`.

The hash comes from `hash_plot_code` in `pyvista/ext/plot_directive.py`, which strips
comments and blank lines and dedents before hashing. So comment edits and reindentation
are free, while renaming a variable, reflowing a call across lines, or reordering
statements produces a new filename for a pixel-identical render. That is the single most
common reason this job fails, and it is always a pure rename.

## Slots that hold several baselines

About forty slots are a **directory** rather than a `.jpg`: `sphx_glr_cell_centers_002/`
holds `0.jpg` and `1.jpg`, `sphx_glr_load_vrml_001_vtksz/` holds `blank.jpg` and
`rendered.jpg`. pytest-pyvista scores the render against every file in the directory and
grades on the closest one, so the slot passes when any single variant matches.

The variants are different pictures, not drifted copies — siblings sit hundreds or
thousands apart, because each covers an environment or an outcome the example can
legitimately produce. Refresh the **closest** variant and leave the others alone;
overwriting them all collapses the slot back to a single accepted render.

## Fetch what the run generated, not what failed

```bash
RUN=$(gh run list -R pyvista/pyvista --workflow docs.yml \
  --branch "$(git branch --show-current)" -L 1 --json databaseId --jq '.[0].databaseId')
gh run download "$RUN" -R pyvista/pyvista --name doc-generated-test-images --dir /tmp/pv-doc-gen
```

`doc-generated-test-images` is every image the build produced, flat, one file per slot.
That is what you want. `doc-failed-test-images` holds only the failures, split across
`errors/`, `warnings/` and `errors_as_warnings/`, each with `from_cache/` and `from_test/`
subdirectories — useful for eyeballing a single diff, hopeless as the source of truth,
because a shifted sequence leaves some slots passing and they simply will not be in it.

Read the job log first to see which examples failed, so you can scope the work:

```bash
gh run view -R pyvista/pyvista --job <job-id> --log-failed | grep -E '^FAILED|Error'
```

## Classify before touching anything

`scripts/match_doc_images.py` does the matching that makes this tractable. Scope it to one
example — `plan` compares every generated image against every cached one, so it refuses
patterns matching more than 80 files.

```bash
python .claude/skills/pyvista-doc-images/scripts/match_doc_images.py \
  plan /tmp/pv-doc-gen --pattern 'sphx_glr_axes_objects_*.jpg'
```

It reports one of four actions per image:

| Action      | Meaning                                                   | What to do                 |
| ----------- | --------------------------------------------------------- | -------------------------- |
| `UNCHANGED` | byte-identical, or re-encoded below the warning threshold | nothing                    |
| `RENAME`    | the same render already exists under another name         | `git mv` it                |
| `REPLACE`   | a cached slot exists and genuinely differs                | copy the generated file in |
| `NEW`       | no cached slot                                            | copy the generated file in |

Rename detection is not just byte equality. A render can be identical and still re-encode
to different bytes, so anything scoring at or below the 200 warning threshold against a
differently-named cached file counts as a rename. Trust that: a value of 74 against
another slot means the picture moved, not that it changed.

Check a `RENAME` against the orphan line before reaching for `git mv`. A real rename leaves
its old name in "cached files the run did not generate"; when the orphan list is empty
every filename is still in use and nothing moved, and the verdict is an example that
renders the same view into several slots. On a directory slot the report names the variant
it matched, and a `REPLACE` says which single variant to overwrite.

## Apply, in this order

Do every `git mv` first, working **down from the highest index**. A sequence that shifted
up by one will overwrite itself if you move `004 → 005` before `005 → 006`.

```bash
git mv tests/doc/doc_image_cache/sphx_glr_axes_objects_009.jpg \
       tests/doc/doc_image_cache/sphx_glr_axes_objects_010.jpg
```

Then copy in the `REPLACE` and `NEW` files from `/tmp/pv-doc-gen`. Then re-run the plan to
confirm nothing is left, and verify every slot against the run:

```bash
python .claude/skills/pyvista-doc-images/scripts/match_doc_images.py \
  verify /tmp/pv-doc-gen --pattern 'sphx_glr_axes_objects_*.jpg'
```

`verify` exits non-zero if any slot exceeds the 500 error threshold. Aim to leave nothing
above 200 either — a slot sitting in the warning band passes today and flakes on the next
unrelated renderer bump.

## Before you accept a REPLACE

`REPLACE` means the picture differs. It does not say your change caused it. A baseline
committed against an older VTK or freetype drifts on its own, most visibly in text
antialiasing, and that drift can sit just under the error threshold for years until a
rename moves it into a slot you are looking at.

Prove which it is by rendering the same scene on your branch and on `main`:

```bash
git checkout origin/main -- <the files you changed>
# render the plot, save it
git checkout HEAD -- <the files you changed>
```

Compare the two with `pv.compare_images`. If they match, your change is innocent and the
cached baseline was stale; refresh it anyway and say so in the commit message rather than
leaving a near-threshold image behind. If they differ, look at the new image and decide
whether the new rendering is what you intended.

Do not regenerate documentation baselines locally on macOS or Windows. The cache is the
Linux CI render, and a local full docs build segfaults on macOS regardless.

## Refreshing drift in bulk

A single artifact is not enough to justify refreshing a slot, because a run contains its
own flaky renders. Download a second `docs.yml` artifact from `main` and refresh only what
both runs agree on.

When the two runs disagree, read `git log <older-sha>..<newer-sha>` before deciding which
one is wrong. A commit that deliberately changes output makes the _older_ run stale rather
than the newer one flaky, and such a commit usually refreshes only the slots it pushed past
500 and leaves its own collateral sitting in the warning band — `_vtksz` twins especially,
whose static sibling was updated without them. Settle it by rendering the example either
side of the suspect commit.

## Traps

| Trap                                      | Why it bites                                                                         |
| ----------------------------------------- | ------------------------------------------------------------------------------------ |
| Copying `errors/from_test/*` wholesale    | replaces renames with fresh bytes and silently misses slots that passed              |
| Moving indices in ascending order         | each move lands on a slot still holding the next image                               |
| Forgetting `_vtksz` and `_thumb`          | they shift with their static sibling and fail on the next run                        |
| Overwriting a whole directory slot        | its variants are different pictures; only the closest one is yours to refresh        |
| Trusting one artifact                     | a run has its own flaky renders, so agreement between two runs is the evidence       |
| Reading a matching `_vtksz` as proof      | judge each file on its own value; the twin says nothing about its static sibling     |
| Committing with `git add -A`              | sweeps in unrelated generated output; stage `tests/doc/doc_image_cache` explicitly   |
| Bumping `sphinx_gallery_thumbnail_number` | the thumbnail follows the plot, so check whether it still points at the same picture |

Git records a shifted sequence as modifications plus one addition, never as renames, since
every filename still exists. That is expected. What matters is that the bytes of an
unchanged render are the bytes already in the repository.
