# CI runners

The documentation cache, build and test jobs in `.github/workflows/docs.yml` run on `blacksmith-8vcpu-ubuntu-2204` for merge-queue, `main`, tag, scheduled and manual runs. A pull request runs them on GitHub-hosted runners unless it carries the `blacksmith` label, which takes effect with its next push. Every other job runs on GitHub-hosted runners.

Caches saved on a Blacksmith runner are invisible to GitHub-hosted jobs, so a job moved there needs its `cache-pyvista-data` producer on the same kind of runner.

| Runner                         | Rate         | Hourly     |
| ------------------------------ | ------------ | ---------- |
| `blacksmith-4vcpu-ubuntu-2204` | `$0.008/min` | `$0.48/hr` |
| `blacksmith-8vcpu-ubuntu-2204` | `$0.016/min` | `$0.96/hr` |

A Blacksmith documentation run is roughly 12 minutes of `blacksmith-8vcpu-ubuntu-2204`, about $0.20.
