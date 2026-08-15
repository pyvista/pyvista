---
name: pyvista-testing
description: Write and repair image regression tests for this repository. Load before touching a plotting test, a baseline image, or an image cache flag.
---

# Image regression testing

`CONTRIBUTING.rst` covers why the caches exist, and `context7.json` covers what makes a
scene worth pinning. This skill is the plumbing: the fixture, the flags, and the two
mistakes that silently disable the check. Reviewing a plotting change belongs in
**pyvista-review**; the local gates and the cost of a red job are in **pyvista-dev**.

## The mechanic

The `verify_image_cache` fixture patches `Plotter.show`. The comparison itself runs from a
`before_close_callback` that `show()` registers on the global theme, and
`show(auto_close=True)` (the default) calls `close()`, which fires the callback, which
screenshots and diffs.

Two consequences, and they cause most of the broken plotting tests here:

- **A regression-tested render must end with `pl.show()`, `mesh.plot()`, or
  `pv.plot(...)`, never a bare `pl.close()`.** A test that builds a plotter and only
  closes it registers no callback, so no comparison runs and the render is untested while
  the test still passes.
- **`pl.close()` is correct only for a plotter you deliberately do not regress**: a
  second plotter in a multi-plotter test, an error-path plotter inside `pytest.raises`, or
  a throwaway used to read a capability.

Multiple `show()` calls in one test produce numbered baselines. `test_foo` writes
`foo.png`, then `foo_1.png`, `foo_2.png`; the `test_` prefix is stripped.

## Where the images live

| Path                          | What                                                    |
| ----------------------------- | ------------------------------------------------------- |
| `tests/plotting/image_cache/` | Committed plotting baselines, PNG, 400 px max dimension |
| `tests/doc/doc_image_cache/`  | Committed documentation baselines, JPG                  |
| `_failed_test_images/`        | Written automatically on failure. Gitignored            |
| `_generated_test_images/`     | Every image the run produced. Gitignored                |

`failed_image_dir`, `generated_image_dir`, `image_cache_dir`, `max_image_size`, and their
`doc_` counterparts are already set in `pyproject.toml` under `[tool.pytest.ini_options]`,
so `make test-plotting` dumps the failures without any extra flag.

Inside a dump, `from_cache/` is the committed baseline and `from_test/` is what the run
rendered. `errors/` exceeded the failure threshold, `warnings/` exceeded only the warning
threshold. To read a large dump, build the HTML report:

```bash
tox run -e image-report -- _failed_test_images _image_report
```

## Accepting or adding a baseline

| Flag                    | Use                                                                      |
| ----------------------- | ------------------------------------------------------------------------ |
| `--add_missing_images`  | A new test with no baseline yet. Writes only what is missing             |
| `--reset_only_failed`   | A render legitimately changed. Overwrites only the baselines that failed |
| `--ignore_image_cache`  | Skip comparison locally while iterating. CI still compares               |
| `--generated_image_dir` | Dump every generated image, not only the failures                        |
| `--reset_image_cache`   | Regenerates **every** baseline the run collected. Maintainer-only        |

Scope every one of these to the node id you mean. Anything passed through `ARGS` replaces
the environment's whole-suite defaults, so the node id below is the entire run:

```bash
make test-plotting ARGS="tests/plotting/test_plotting.py::test_my_render --reset_only_failed"
```

`--reset_image_cache` launders unrelated regressions into the cache and leaves the suite
comparing new renders against themselves. It exists for a deliberate whole-cache
regeneration at a release. Do not reach for it to clear a red job, and never run it
unscoped.

Look at the images before you commit them. A failed image test is sometimes a real
regression, and `git diff` cannot tell you which.

## Take baselines from CI, not from your GPU

Baselines are the Linux CI renders. Your driver produces slightly different pixels, so
regenerating locally on macOS or Windows bakes platform-specific output into the canonical
cache.

The failing job uploads what it rendered: `failed_test_images-<job>-<matrix>` from
`Unit Testing and Deployment`, `doc-failed-test-images` from `Build Documentation`.

```bash
RUN=$(gh run list -R pyvista/pyvista --workflow testing-and-deployment.yml \
  --branch "$(git branch --show-current)" -L 1 --json databaseId --jq '.[0].databaseId')
gh run download "$RUN" -R pyvista/pyvista --pattern 'failed_test_images-*' --dir /tmp/pv-failed
ls /tmp/pv-failed          # pick a Linux leg; macOS and Windows renders differ
find /tmp/pv-failed/failed_test_images-Linux-* -path '*/errors/from_test/*.png' \
  -exec cp {} tests/plotting/image_cache/ \;
git diff --stat tests/plotting/image_cache/
```

Look at every image the copy touched before committing. Documentation baselines work the
same way with `doc-failed-test-images`, `docs.yml`, and `tests/doc/doc_image_cache/`,
where the files are `.jpg`. Copying from a non-Linux leg breaks Linux CI on the next run.

## What the scene has to show

A centered `pv.Sphere()` from an axis-aligned camera looks the same under almost any
rendering change, so it passes whether the code is right or broken. `context7.json` states
the rule: asymmetric geometry such as `examples.load_random_hills()` or a `download_*`
dataset, an off-axis camera (`pl.camera_position = 'iso'`), a distinctive colormap, and
`show_edges=True` when the surface is too smooth to read.

Assert the behavior numerically before the render. The image catches what the numbers
miss; it is a poor substitute for them.

```python
import pyvista as pv
from pyvista import examples


def test_warp_by_scalar_displaces_surface(verify_image_cache):
    mesh = examples.load_random_hills()
    warped = mesh.warp_by_scalar(factor=2.0)
    assert warped.bounds.z_max > mesh.bounds.z_max  # behavior first
    pl = pv.Plotter()
    pl.add_mesh(warped, cmap='coolwarm', show_edges=True)
    pl.camera_position = 'iso'
    pl.show()  # the capture and compare; not pl.close()
```

## Per-test knobs

Set these on the fixture object at the top of the test body. Thresholds are attributes,
not command line flags.

| Attribute                       | Effect                                                        |
| ------------------------------- | ------------------------------------------------------------- |
| `skip = True`                   | Render, compare nothing, on every platform                    |
| `windows_skip_image_cache`      | Skip the comparison on Windows only                           |
| `macos_skip_image_cache`        | Skip the comparison on macOS only                             |
| `high_variance_test = True`     | Swap the 500/200 error/warning thresholds for 1000/1000       |
| `error_value` / `warning_value` | Per-test thresholds. Prefer `high_variance_test`              |
| `allow_useless_fixture`         | Opt out of the "fixture used but no images generated" failure |

Reach for a per-platform skip or a looser threshold only when a render legitimately
diverges. Making the scene deterministic is the better fix, and neither knob justifies a
second cache directory: a test with several accepted renders gets a sub-directory of
baselines under `image_cache/<test>/` instead, as `plot_show_grid` already does. The test
image is compared against each accepted baseline in turn and passes if any one matches.

## Behavioral tests under the autouse fixture

`tests/plotting/test_plotting.py` requests `verify_image_cache` for every test through an
autouse wrapper, so a passing test that never rendered fails with _"Fixture
`verify_image_cache` is used but no images were generated"_. That guard is there to catch
a plotting test that forgot `show()`. A genuinely behavioral test opts out through the
`no_images_to_verify` fixture already defined in that module, which sets
`allow_useless_fixture` and asserts `n_calls == 0` afterwards.

Do not request the fixture at all in a test that asserts return values, mesh metadata, IO
round trips, or a raise path. Those belong in `tests/core/`.

## Documentation images

The documentation build has its own cache, its own flags (`--doc_mode`,
`--doc_images_dir`), and its own tox environment, `docs-test-images`. Editing a docstring
example that renders changes a documentation baseline, so a `Build Documentation` failure
after a docstring edit is usually this and not a broken build. Fetch
`doc-failed-test-images` from the run and treat it the same way as a plotting baseline.
