---
name: pyvista-review
description: Review a PyVista change against the project's actual merge bar. Use before opening a pull request and whenever asked to review a branch, diff, or pull request.
---

# Reviewing a PyVista change

Run this from a **subagent** when reviewing your own work, so the reviewer has to derive
intent from the diff alone.

## The standard

From [Google's _Standard of Code Review_](https://google.github.io/eng-practices/review/reviewer/standard.html),
because an adversarial reviewer without a stopping rule blocks everything:

> In general, reviewers should favor approving a CL once it is in a state where it
> definitely improves the overall code health of the system being worked on, even if the
> CL isn't perfect.

Weigh the improvement you are proposing against the cost of another round trip. Two
corollaries:

- **Principles, not preferences.** If `ruff`, `numpydoc`, `CONTRIBUTING.rst`, or
  `context7.json` settles it, it is settled. If it is genuinely your taste, say so and
  mark it non-blocking.
- **Consistency loses to the guide but beats your taste.** Matching the surrounding
  module is the tie-breaker when nothing else decides.

## Review what the diff implies, not only the diff

The findings worth the most come from leaving the changed lines. Each of these is a
command, not an intention:

- **Look at the rendered documentation.** `make docs` builds it locally. Pull requests
  from the main repository get a Netlify preview; pull requests from forks attach the
  HTML as an artifact on the `Build Documentation` job. Broken images, missing gallery
  entries, and removed `__doc__` attributes are only visible here.
- **Open the failing job and read the assertion.** Not the job name, the assertion.
- **`grep` for the constant, helper, or alias being introduced.** It usually exists.
- **`git log -S"<symbol>"`.** If this name has already been changed once, changing it
  again is churn, and changing it back is a regression of a prior pull request.
- **Ask what input would make each new test pass while the code is wrong.**

## Checklist

Google's review dimensions. Read every changed line; what you cannot follow, the next
reader will not follow either.

| Dimension     | Ask                                                               |
| ------------- | ----------------------------------------------------------------- |
| Tests         | Are they correct and useful? Will they fail when the code breaks? |
| Complexity    | Could it be simpler? Is it more generic than it needs to be?      |
| Design        | Does this belong in PyVista, and does it fit the rest of the API? |
| Functionality | Does it do what the author intended, edge cases included?         |
| Documentation | Directives, examples, and pages that actually render              |
| Naming        | Does the name say what the thing is, without being long?          |
| Consistency   | Does it match the surrounding module?                             |
| Comments      | Do they explain why, rather than what?                            |
| Style         | Automated. Prefix anything left over with `nit:`.                 |

Tests and complexity lead because they are what this repository sends back most often.
On complexity, Google's phrasing describes the most common defect in generated code:

> developers have made the code more generic than it needs to be, or added functionality
> that isn't presently needed

Speculative generality is a finding: an abstraction with one caller, a context manager
where an assignment would do, a configuration knob nobody asked for.

## What this repository weights heavily

**Tests, above everything.** The criticism is rarely "add a test" and usually "this test
would pass anyway". Look for the fixture that makes the assertion true regardless, the
uniform input that never reaches the branch under test, the missing negative case, and
the assertion that checks one attribute where a round trip was available. A commented-out
case in a `parametrize` list is always challenged. For plotting changes, load
**pyvista-testing** and check two things the diff hides: symmetric geometry renders
identically whether the code is right or broken, and a plotter closed without `show()`
never compares anything at all.

**Defaults.** Whether a flag should be automatic rather than opt-in, and whether the
default is what a user wants. The argument that lands is about the mistake someone will
make later, not present correctness.

**Error messages.** Read them as a user who has just hit one. Vague or inaccurate text is
a finding.

**Version directives.** `versionadded`, `versionchanged`, and `deprecated` on anything
new or changed, with the right version.

**Reuse.** Constants, `Literal` aliases, capability probes, and test guards should be
defined once and imported. A re-derived version check where a shared constant exists is a
certain comment.

**Deprecation discipline.** Nothing pre-existing is removed or silently changed, private
helpers included. Watch for a name being changed for the second time; a rename followed
by a rename back has cost downstream users here before.

**Scope.** Ask for a split when the change contains genuinely separable work, such as a
refactor riding along with a feature. Do not ask on size alone. Some changes must land
atomically, and `CONTRIBUTING.rst` weighs churn against us too.

## Writing the comments

Inline comments here are short, and roughly a quarter carry a suggestion block. Brevity
and applicable fixes, not essays. Regenerate the distribution with:

```bash
gh api --paginate 'repos/pyvista/pyvista/pulls/comments?per_page=100' --jq '.[].body'
```

Label findings with [Conventional Comments](https://conventionalcomments.org/) so the
author can tell a defect from a preference:

| Label         | For                                                                                   | Blocking       |
| ------------- | ------------------------------------------------------------------------------------- | -------------- |
| `issue:`      | A defect. Wrong output, broken edge case, invalid state reachable from the public API | Yes            |
| `suggestion:` | A concrete improvement, with the alternative named                                    | Say which      |
| `question:`   | A suspected problem you are unsure of, often "is there a test for that?"              | Until answered |
| `nit:`        | Trivial preference                                                                    | Never          |
| `thought:`    | An idea worth recording, no action expected                                           | Never          |
| `praise:`     | Something done well                                                                   | Never          |

Style, following Google's guidance on
[writing review comments](https://google.github.io/eng-practices/review/reviewer/comments.html):

- Comment on the code, never on the developer.
- Frame requests collectively and as questions. "Can we define this once?" rather than an
  instruction.
- Explain why. A request without a reason gets argued about instead of applied.
- Write a suggestion block whenever the fix is mechanical.
- Cite claims. Link the earlier pull request, the VTK issue, the failing job, the
  documentation preview.
- Say when you are unsure. Hedging costs nothing.
- Thank first-time contributors, then go straight to the substance.

One rule worth enforcing strictly:

> Explanations written only in the code review tool are not helpful to future code
> readers.

If the author has to explain the code in a reply, the fix belongs in a comment or a
clearer name, not in the reply.

## Output

Group by file, most severe first. Close with what must change before merge, what could be
a follow-up, and what you checked and found sound. Confirmations are worth stating: they
tell the author which of their decisions survived scrutiny.

Avoid praise padding, rubber-stamping without evidence, vague requests that name no
alternative, and restating anything the linters already enforce.
