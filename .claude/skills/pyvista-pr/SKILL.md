---
name: pyvista-pr
description: Write and open a PyVista pull request in the project's own style. Use whenever drafting or updating a pull request title or body for this repository.
---

# Writing a PyVista pull request

Do not open a pull request whose review findings are still outstanding.

## Never hard-wrap the body

Write each paragraph as one long line and let GitHub wrap it. Hard-wrapped bodies break
when anyone edits or quotes them, and leave stray breaks mid-sentence. This is the one
place the wrapping used in source files is wrong.

## Length

Pull request descriptions here are short. The median merged description is a paragraph;
three quarters are under about a thousand characters, and the ones above that earn their
length with screenshots and rendered previews rather than with prose.

Language models write far past that unprompted, and structure it into sections nobody
here uses. If your draft has a Motivation heading, an Implementation heading, and a
Testing heading, it is wrong for this repository regardless of how accurate it is.

Two or three paragraphs is the ceiling. Cut to roughly a fifth of what feels complete:
the first draft is reliably five times too long. Anything a reviewer can read in the
diff, work out from the linked issue, or does not need in order to review is padding —
including justifications for the shape of the pull request, follow-up work you chose not
to do, and summaries of your own process.

Regenerate the distribution with:

```bash
gh pr list -R pyvista/pyvista --state merged --limit 400 \
  --json body,title,author --jq '.[] | select(.author.is_bot | not) | .body | length'
```

## Shape

`.github/PULL_REQUEST_TEMPLATE.md` supplies two headings, `### Overview` and
`### Details`. Write the overview; drop `### Details` unless there is a genuine list to
put under it. Do not invent other headings. The only ones that recur are labels around
before-and-after screenshots.

The template also states the conventions inline: link related issues and pull requests,
and use the keyword `resolves` in front of an issue number when the change fully closes
it.

Link rather than explain. Point at the issue, the failing job, the discussion comment, or
the documentation preview. A link carries more than a paragraph describing the same
thing, and it lets a reviewer verify the claim.

Bullets are for a genuine list, most often secondary changes that came along with the
main one. Do not use them to break a paragraph into fragments.

Screenshots for anything visual, labeled so the pair is readable:

```markdown
Main:
<image>

This branch:
<image>
```

## Titles

Imperative, sentence case, identifiers in backticks, no trailing period, around fifty
characters. Common openers are Add, Fix, Use, Bump, Remove, Make, Move, Enable,
Deprecate, and Support. A `DOC:` or `ci:` prefix appears occasionally and is optional.
Four merged examples:

```
Add `preserve_aspect_ratio` to `resize` filter
Extend `wrap()`'s fast validation path to composite datasets
Fix errors with vtk-master tests
Default COVERAGE_CORE to ctrace when unset
```

## Voice

Plain and direct. First person is fine. Contractions are fine. Descriptions here read as
working notes to colleagues, not as release announcements.

Say why the change exists and what it does. Do not narrate the diff. Anything a reviewer
can read in the diff does not belong in the description.

Avoid: headings the template does not have, bold-label bullets, summary tables of what
changed, checklists, emoji, and the register of "this pull request introduces a
comprehensive refactor". Avoid a Testing section listing what you ran; if a specific
verification matters, it fits in a clause.

## Disclosing AI assistance

The `Generative AI` section of `CONTRIBUTING.rst` adopts the Python Developer's Guide
[policy on AI tools](https://devguide.python.org/getting-started/ai-tools/): disclosure in
the pull request description is appreciated and not required, and the person opening the
pull request owns everything in it. When disclosing, use one inline clause naming what the tool did and confirming the
author understood it, in the form merged descriptions already use:

> Changes drafted by Claude Opus 5 but fully understood by me

No banner, no badge, no separate heading, no generated-with footer. Put it in the
description rather than only in a commit trailer, because the description is what a
reviewer reads first.

## Labels that start extra CI

Four labels each start a job a pull request does not otherwise get. They cost real runner
time, so ask for one because the change touches what it covers, not by default:

| Label                 | Starts                                      | Ask for it when                                                                         |
| --------------------- | ------------------------------------------- | --------------------------------------------------------------------------------------- |
| `vtk-dev-testing`     | the suite against VTK development wheels    | anything version-dependent: wrapping, object lifetimes, a VTK behavior that has changed |
| `vtk-master-testing`  | a VTK build from master, then the suite     | the change depends on unreleased VTK, or the dev wheels are not recent enough           |
| `integration-testing` | mne, trame, pyvistaqt, geovista, playwright | public API behavior, object lifetimes, plotting defaults -- what downstream sits on     |
| `docker`              | the Docker image build                      | packaging, or a dependency the image installs                                           |

A label only takes effect on the next run, so it goes on before the final push, or the
branch gets pushed again afterwards; `CONTRIBUTING.rst` says the same for the VTK labels.
You cannot apply labels yourself, so name the ones the change warrants when you hand the
pull request over.

## Before opening

1. Draft the title and body.
2. If the body is longer than the diff deserves and carries no screenshots, cut rather
   than restructure.
3. Confirm every paragraph is a single unbroken line.
4. Strip anything that reads as generated: section scaffolding, bold labels, hedged
   summary sentences, rule-of-three lists.
5. Add `Resolves #NNNN` when the change fully closes an issue.
6. Confirm the local gates pass. Continuous integration runs on `pull_request`, so a red
   pull request costs the whole matrix.
7. Open as a draft unless the change is ready for review.

## Example

Adapted from a real description. Instead of this:

```markdown
## Summary

This PR introduces a comprehensive refactor of the cell array accessors to provide a
consistent, first-class API across all dataset types.

### Motivation

The existing API is inconsistent across classes...

### Implementation

- **`PolyData`**: added `cell_offsets` and `cell_connectivity`
- **`UnstructuredGrid`**: deprecated `offset`

### Testing

Added 14 new tests covering read-only enforcement...
```

write this:

```markdown
### Overview

Resolves #8909.

`vtkCellArray` has stored offsets and connectivity separately since VTK 9, but we expose them under three different names depending on the class, and only some are read-only. This adds `cell_offsets` and `cell_connectivity` everywhere and deprecates the rest.

Plain `offsets` and `connectivity` were not available, i.e. `connectivity` shadows the `DataSetFilters.connectivity` filter.
```

Note the unbroken paragraphs.
