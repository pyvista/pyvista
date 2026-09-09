# CI runners

Every job runs on GitHub-hosted runners by default. The `blacksmith` label on a pull request moves the documentation cache, build and test jobs in `.github/workflows/docs.yml` to `blacksmith-8vcpu-ubuntu-2204`, starting with the pull request's next run. Merge-queue, `main`, tag and scheduled runs never use it.

Caches saved on a Blacksmith runner are invisible to GitHub-hosted jobs, so a job moved there needs its `cache-pyvista-data` producer on the same kind of runner.

| Runner                         | Rate         | Hourly     |
| ------------------------------ | ------------ | ---------- |
| `blacksmith-4vcpu-ubuntu-2204` | `$0.008/min` | `$0.48/hr` |
| `blacksmith-8vcpu-ubuntu-2204` | `$0.016/min` | `$0.96/hr` |

A labelled documentation run is roughly 12 minutes of `blacksmith-8vcpu-ubuntu-2204`, about $0.20.
