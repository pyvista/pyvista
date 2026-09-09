# CI runners

Every job runs on GitHub-hosted runners.

## Blacksmith

A Linux job can be moved to Blacksmith by replacing its `runs-on` label in the workflow file. Caches saved on a Blacksmith runner are invisible to GitHub-hosted jobs, so a job moved there needs its `cache-pyvista-data` producer on the same kind of runner.

| Runner                         | Rate         | Hourly     |
| ------------------------------ | ------------ | ---------- |
| `blacksmith-4vcpu-ubuntu-2204` | `$0.008/min` | `$0.48/hr` |
| `blacksmith-8vcpu-ubuntu-2204` | `$0.016/min` | `$0.96/hr` |
