# CI runners

PyVista uses Blacksmith for the Ubuntu jobs that still depended on the retired Linux self-hosted runner.

| Workflow                                | Job(s)                                               | Runner                         |
| --------------------------------------- | ---------------------------------------------------- | ------------------------------ |
| `.github/workflows/docs.yml`            | `cache-pyvista-data`, `doc`                          | `blacksmith-8vcpu-ubuntu-2204` |
| `.github/workflows/style-docstring.yml` | Ubuntu `cache-pyvista-data`, Ubuntu `docstringcheck` | `blacksmith-4vcpu-ubuntu-2204` |

All other Linux workflows stay on GitHub-hosted Ubuntu for now, and the macOS matrix stays on the existing self-hosted runner.

## Budget

| Runner                         | Rate         | Hourly     |
| ------------------------------ | ------------ | ---------- |
| `blacksmith-4vcpu-ubuntu-2204` | `$0.008/min` | `$0.48/hr` |
| `blacksmith-8vcpu-ubuntu-2204` | `$0.016/min` | `$0.96/hr` |

| Job                     | Runner                         | Expected wall | Estimated cost per run |
| ----------------------- | ------------------------------ | ------------- | ---------------------- |
| Ubuntu `docstringcheck` | `blacksmith-4vcpu-ubuntu-2204` | 7-10 min      | `$0.056-0.080`         |
| `Build Documentation`   | `blacksmith-8vcpu-ubuntu-2204` | 10-15 min     | `$0.16-0.24`           |

Based on recent PR volume and the sampled docs/docstring runtimes, the current Blacksmith footprint should stay roughly within **$15-40/month**.

## Emergency rollback

Blacksmith is a label-only migration on Ubuntu. To switch a job back to GitHub-hosted runners, replace the `blacksmith-*` label with the prior `ubuntu-22.04` or `ubuntu-latest` label in the workflow file and rerun CI.
