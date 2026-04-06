"""Generate a static HTML report for failed/differing test images.

Usage:
    python generate_test_image_report.py <input_dir> <output_dir>

Where:
    input_dir:  Directory containing downloaded ``failed_test_images-*`` artifacts.
    output_dir: Directory to write the HTML report and copied images to.

Exit codes:
    0 - Report generated (or no differences found).
    1 - Usage error.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

from rich.console import Console
from rich.table import Table

CATEGORIES = [
    ('errors', 'Errors (Test Failures)', 'error'),
    ('errors_as_warnings', 'Errors as Warnings', 'warning-error'),
    ('warnings', 'Warnings', 'warning'),
]


def _is_raw_failed_dir(path: Path) -> bool:
    """Return True if *path* looks like a ``_failed_test_images`` dir (not an artifact wrapper)."""
    return any((path / cat).is_dir() for cat, _, _ in CATEGORIES)


def _collect_image_pairs(input_dir: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Walk artifact directories and return structured image-comparison data.

    Handles two layouts:

    * **CI layout** -- ``input_dir`` contains ``failed_test_images-*`` subdirs.
    * **Local layout** -- ``input_dir`` *is* the ``_failed_test_images`` dir
      (contains ``errors/``, ``warnings/``, etc. directly).

    Returns
    -------
    dict
        ``{job_name: {category: [{"test_name", "from_test", "from_cache"}, ...]}}``

    """
    # Detect local layout: the input dir itself is a _failed_test_images dir
    if _is_raw_failed_dir(input_dir):
        artifact_dirs = [input_dir]
    else:
        artifact_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())

    results: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for artifact_dir in artifact_dirs:
        if not artifact_dir.is_dir():
            continue

        job_name = artifact_dir.name.removeprefix('failed_test_images-')
        job_data: dict[str, list[dict[str, Any]]] = {cat: [] for cat, _, _ in CATEGORIES}

        for category, _, _ in CATEGORIES:
            cat_dir = artifact_dir / category
            from_test_dir = cat_dir / 'from_test'
            from_cache_dir = cat_dir / 'from_cache'

            if not from_test_dir.is_dir():
                continue

            for test_img in sorted(from_test_dir.iterdir()):
                if not test_img.is_file():
                    continue

                test_name = test_img.stem
                cache_images: list[Path] = []

                # Flat file match (errors / warnings)
                cache_flat = from_cache_dir / test_img.name
                if cache_flat.is_file():
                    cache_images.append(cache_flat)

                # Subdirectory match (errors_as_warnings stores multiple cached variants)
                cache_subdir = from_cache_dir / test_name
                if cache_subdir.is_dir():
                    cache_images.extend(sorted(cache_subdir.iterdir()))

                job_data[category].append(
                    {
                        'test_name': test_name,
                        'from_test': test_img,
                        'from_cache': [p for p in cache_images if p.is_file()],
                    }
                )

        if any(job_data[cat] for cat, _, _ in CATEGORIES):
            results[job_name] = job_data

    return results


def _safe_dirname(name: str) -> str:
    return (
        name.replace('/', '_')
        .replace(' ', '_')
        .replace('(', '')
        .replace(')', '')
        .replace('[', '')
        .replace(']', '')
    )


_CSS = """\
:root{
--bg:#fff;--bg-surface:#f6f8fa;--bg-overlay:#fff;
--fg:#1f2328;--fg-muted:#656d76;--fg-heading:#1f2328;
--border:#d0d7de;--border-muted:#d8dee4;
--link:#0969da;
--err:#cf222e;--err-bg:#ffebe9;--err-fg:#cf222e;
--warn-err:#9a6700;--warn-err-bg:#fff8c5;
--warn:#0969da;--warn-bg:#ddf4ff;--warn-fg:#0969da;
--ok:#1a7f37;
--nav-hover:#f3f4f6}
@media(prefers-color-scheme:dark){:root{
--bg:#0d1117;--bg-surface:#161b22;--bg-overlay:#0d1117;
--fg:#c9d1d9;--fg-muted:#8b949e;--fg-heading:#f0f6fc;
--border:#30363d;--border-muted:#21262d;
--link:#58a6ff;
--err:#da3633;--err-bg:#1c1214;--err-fg:#f85149;
--warn-err:#d29922;--warn-err-bg:#1c1a14;
--warn:#58a6ff;--warn-bg:#121d2f;--warn-fg:#58a6ff;
--ok:#238636;
--nav-hover:#161b22}}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
background:var(--bg);color:var(--fg);padding:20px}
.header{text-align:center;padding:20px 0 30px;margin-bottom:10px}
.header h1{font-size:1.8em;margin-bottom:10px;color:var(--fg-heading)}
.context{margin-bottom:8px;font-size:.9em;color:var(--fg-muted)}
.context a{color:var(--link)}
.context code{background:var(--bg-surface);
padding:2px 6px;border-radius:4px;font-size:.85em}
.nav{position:sticky;top:0;background:var(--bg);
border-bottom:1px solid var(--border-muted);
padding:10px 0;z-index:100;margin-bottom:20px}
.nav-row{display:flex;gap:8px;flex-wrap:wrap;
justify-content:center;align-items:center}
.nav-row+.nav-row{margin-top:6px}
.nav-row a{color:var(--link);text-decoration:none;
padding:4px 10px;border-radius:4px;font-size:.85em}
.nav-row a:hover{background:var(--nav-hover)}
.badge{padding:6px 14px;border-radius:20px;font-size:.9em;
font-weight:600;cursor:pointer;user-select:none;
border:2px solid transparent;transition:opacity .15s}
.badge.error{background:var(--err);color:#fff}
.badge.warning-error{background:var(--warn-err);color:#fff}
.badge.warning{background:var(--warn);color:#fff}
.badge.ok{background:var(--ok);color:#fff}
.badge.inactive{opacity:.35}
.job{margin-bottom:40px}
.job-header{font-size:1.4em;padding:12px 16px;
background:var(--bg-surface);border:1px solid var(--border);
border-radius:8px 8px 0 0;color:var(--fg-heading)}
.category{margin-bottom:20px}
.category-header{font-size:1.1em;padding:8px 16px;
border-left:4px solid;margin:0}
.category-header.error{border-color:var(--err);
background:var(--err-bg);color:var(--err-fg)}
.category-header.warning-error{border-color:var(--warn-err);
background:var(--warn-err-bg);color:var(--warn-err)}
.category-header.warning{border-color:var(--warn);
background:var(--warn-bg);color:var(--warn-fg)}
.card{border:1px solid var(--border);border-radius:8px;
margin:12px 0;padding:16px;background:var(--bg-surface)}
.card.error{border-left:4px solid var(--err)}
.card.warning-error{border-left:4px solid var(--warn-err)}
.card.warning{border-left:4px solid var(--warn)}
.card h3{font-size:1em;margin-bottom:12px;
color:var(--fg-heading);word-break:break-all}
.comparison{display:flex;gap:16px;flex-wrap:wrap}
.image-col{flex:1;min-width:300px}
.image-col h4{font-size:.85em;color:var(--fg-muted);margin-bottom:8px}
.cache-label{font-weight:normal}
.image-col img{width:100%;border:1px solid var(--border);
border-radius:4px;background:var(--bg-overlay)}
.no-image{padding:40px;text-align:center;
border:1px dashed var(--border);border-radius:4px;color:var(--fg-muted)}
.hidden{display:none}
"""

_JS = """\
<script>
document.addEventListener('DOMContentLoaded',()=>{
  const badges=document.querySelectorAll('.badge[data-cat]');
  const active=new Set(Array.from(badges).map(b=>b.dataset.cat));
  function apply(){
    document.querySelectorAll('.category[data-cat]').forEach(el=>{
      el.classList.toggle('hidden',!active.has(el.dataset.cat));
    });
    document.querySelectorAll('.job').forEach(job=>{
      const vis=job.querySelectorAll('.category:not(.hidden)');
      job.classList.toggle('hidden',vis.length===0);
    });
    badges.forEach(b=>b.classList.toggle('inactive',!active.has(b.dataset.cat)));
  }
  badges.forEach(b=>b.addEventListener('click',()=>{
    const c=b.dataset.cat;
    if(active.has(c))active.delete(c);else active.add(c);
    apply();
  }));
});
</script>"""


def _build_card_html(entry: dict[str, Any], css_class: str) -> str:
    """Build a single image-comparison card."""
    name = html.escape(entry['test_name'])
    if entry['cache_rels']:
        cache_cols = ''
        for cr in entry['cache_rels']:
            label = html.escape(cr['label'])
            cache_cols += (
                '<div class="image-col">'
                f'<h4>Baseline <span class="cache-label">({label})</span></h4>'
                f'<img src="{cr["path"]}" alt="Baseline: {name}" loading="lazy">'
                '</div>'
            )
    else:
        cache_cols = (
            '<div class="image-col">'
            '<h4>Baseline</h4>'
            '<div class="no-image">No baseline image found</div>'
            '</div>'
        )

    return (
        f'<div class="card {css_class}">'
        f'<h3>{name}</h3>'
        f'<div class="comparison">{cache_cols}'
        f'<div class="image-col"><h4>Generated (Test)</h4>'
        f'<img src="{entry["test_rel"]}" alt="Generated: {name}" loading="lazy">'
        '</div></div></div>'
    )


def _build_html(
    results: dict[str, dict[str, list[dict[str, Any]]]],
    summary: dict[str, int],
    pr_url: str = '',
    branch: str = '',
) -> str:
    """Return the full HTML page as a string."""
    job_sections = []
    for job_name, job_data in results.items():
        anchor = _safe_dirname(job_name)
        safe_job = html.escape(job_name)
        sections = ''
        for category, label, css_class in CATEGORIES:
            entries = job_data.get(category, [])
            if not entries:
                continue
            cards = ''.join(_build_card_html(e, css_class) for e in entries)
            sections += (
                f'<div class="category" data-cat="{category}">'
                f'<h2 class="category-header {css_class}">{label} ({len(entries)})</h2>'
                f'{cards}</div>'
            )
        job_sections.append(
            f'<div class="job" id="{anchor}">'
            f'<h1 class="job-header">{safe_job}</h1>'
            f'{sections}</div>'
        )

    # Filter badges
    badges = []
    if summary['num_errors']:
        badges.append(
            f'<span class="badge error" data-cat="errors">{summary["num_errors"]} errors</span>'
        )
    if summary['num_errors_as_warnings']:
        badges.append(
            f'<span class="badge warning-error" data-cat="errors_as_warnings">'
            f'{summary["num_errors_as_warnings"]} errors as warnings</span>'
        )
    if summary['num_warnings']:
        badges.append(
            f'<span class="badge warning" data-cat="warnings">'
            f'{summary["num_warnings"]} warnings</span>'
        )
    badge_html = ' '.join(badges)

    nav_links = ''.join(f'<a href="#{_safe_dirname(j)}">{html.escape(j)}</a>' for j in results)

    diff_s = 's' if summary['total_differences'] != 1 else ''
    job_s = 's' if summary['num_jobs'] != 1 else ''
    subtitle = (
        f'{summary["total_differences"]} image difference{diff_s}'
        f' across {summary["num_jobs"]} test job{job_s}'
    )

    # Build context line (PR link + branch)
    context_parts = []
    if pr_url:
        safe_url = html.escape(pr_url, quote=True)
        context_parts.append(f'<a href="{safe_url}">Pull Request</a>')
    if branch:
        context_parts.append(f'<code>{html.escape(branch)}</code>')
    context_line = f'<p class="context">{" | ".join(context_parts)}</p>\n' if context_parts else ''

    content = ''.join(job_sections)

    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, '
        'initial-scale=1.0">\n'
        '<title>PyVista Test Image Report</title>\n'
        f'<style>\n{_CSS}</style>\n'
        '</head>\n<body>\n'
        '<div class="header">\n'
        '<h1>PyVista Test Image Report</h1>\n'
        f'{context_line}'
        f'<p>{subtitle}</p>\n'
        '</div>\n'
        f'<nav class="nav">'
        f'<div class="nav-row">{badge_html}</div>\n'
        f'<div class="nav-row">{nav_links}</div>'
        f'</nav>\n'
        f'{content}\n'
        f'{_JS}\n'
        '</body>\n</html>'
    )


def _compute_summary(
    results: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, int]:
    return {
        'num_jobs': len(results),
        'total_differences': sum(
            len(entries) for job_data in results.values() for entries in job_data.values()
        ),
        'num_errors': sum(len(job_data.get('errors', [])) for job_data in results.values()),
        'num_warnings': sum(len(job_data.get('warnings', [])) for job_data in results.values()),
        'num_errors_as_warnings': sum(
            len(job_data.get('errors_as_warnings', [])) for job_data in results.values()
        ),
    }


def generate_report(
    input_dir: Path,
    output_dir: Path,
    pr_url: str = '',
    branch: str = '',
) -> dict[str, int] | None:
    """Generate the full report. Returns summary dict, or ``None`` if nothing to report."""
    results = _collect_image_pairs(input_dir)
    if not results:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / 'images'
    images_dir.mkdir(exist_ok=True)

    # Copy images into output and annotate entries with relative paths
    for job_name, job_data in results.items():
        job_dir = images_dir / _safe_dirname(job_name)
        job_dir.mkdir(exist_ok=True)
        for category, _, _ in CATEGORIES:
            for entry in job_data.get(category, []):
                # Test image
                dest = job_dir / f'{category}_test_{entry["test_name"]}{entry["from_test"].suffix}'
                shutil.copy2(entry['from_test'], dest)
                entry['test_rel'] = dest.relative_to(output_dir).as_posix()

                # Cache (baseline) images
                entry['cache_rels'] = []
                for i, cache_img in enumerate(entry['from_cache']):
                    suffix = f'_{i}' if len(entry['from_cache']) > 1 else ''
                    dest = (
                        job_dir
                        / f'{category}_cache_{entry["test_name"]}{suffix}{cache_img.suffix}'
                    )
                    shutil.copy2(cache_img, dest)
                    label = cache_img.name if len(entry['from_cache']) > 1 else 'Baseline'
                    entry['cache_rels'].append(
                        {'path': dest.relative_to(output_dir).as_posix(), 'label': label}
                    )

    summary = _compute_summary(results)

    # Write summary JSON (consumed by CI for the PR comment)
    (output_dir / 'summary.json').write_text(json.dumps(summary))

    # Write HTML
    page = _build_html(results, summary, pr_url=pr_url, branch=branch)
    (output_dir / 'index.html').write_text(page)

    return summary


def _print_summary(summary: dict[str, int], output_dir: Path) -> None:
    console = Console()
    table = Table(title='Test Image Report', show_header=False, border_style='dim')
    table.add_column(style='bold')
    table.add_column(justify='right')
    table.add_row('Output', str(output_dir / 'index.html'))
    table.add_row('Jobs', str(summary['num_jobs']))
    table.add_row('Total differences', str(summary['total_differences']))
    if summary['num_errors']:
        table.add_row('[red]Errors[/]', str(summary['num_errors']))
    if summary['num_errors_as_warnings']:
        table.add_row('[yellow]Errors as warnings[/]', str(summary['num_errors_as_warnings']))
    if summary['num_warnings']:
        table.add_row('[blue]Warnings[/]', str(summary['num_warnings']))
    console.print(table)


def main() -> None:
    """CLI entry point for generating the test image report.

    Reads optional ``PR_URL`` and ``BRANCH`` environment variables
    to embed links in the report header.
    """
    if len(sys.argv) != 3:
        sys.stdout.write(f'Usage: {sys.argv[0]} <input_dir> <output_dir>\n')
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not input_dir.is_dir():
        sys.stdout.write(f'Input directory does not exist: {input_dir}\nNo artifacts to report.\n')
        sys.exit(0)

    pr_url = os.environ.get('PR_URL', '')
    branch = os.environ.get('BRANCH', '')

    summary = generate_report(
        input_dir,
        output_dir,
        pr_url=pr_url,
        branch=branch,
    )

    if summary is None:
        sys.stdout.write('No image differences found. Skipping report generation.\n')
        sys.exit(0)

    _print_summary(summary, output_dir)


if __name__ == '__main__':
    main()
