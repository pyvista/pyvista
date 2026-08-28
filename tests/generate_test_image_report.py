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

import html as html_mod
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

from rich.console import Console
from rich.table import Table

CATEGORIES = [
    ('errors', 'Errors', 'error'),
    ('errors_as_warnings', 'Errors as Warnings', 'warning-error'),
    ('warnings', 'Warnings', 'warning'),
]

CAT_KEYS = [c[0] for c in CATEGORIES]


def _is_raw_failed_dir(path: Path) -> bool:
    """Return True if *path* looks like a ``_failed_test_images`` dir (not an artifact wrapper)."""
    return any((path / cat).is_dir() for cat in CAT_KEYS)


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
    if _is_raw_failed_dir(input_dir):
        artifact_dirs = [input_dir]
    else:
        artifact_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())

    results: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for artifact_dir in artifact_dirs:
        if not artifact_dir.is_dir():
            continue

        job_name = artifact_dir.name.removeprefix('failed_test_images-')
        job_data: dict[str, list[dict[str, Any]]] = {cat: [] for cat in CAT_KEYS}

        for category in CAT_KEYS:
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

                cache_flat = from_cache_dir / test_img.name
                if cache_flat.is_file():
                    cache_images.append(cache_flat)

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

        if any(job_data[cat] for cat in CAT_KEYS):
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


def _pivot_by_test(
    results: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Pivot from {job: {cat: [entries]}} to {test_name: {cat: [{...job info}]}}.

    Each entry in the output list has the job_name added and
    the original entry data preserved.
    """
    pivoted: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for job_name, job_data in results.items():
        for category in CAT_KEYS:
            for entry in job_data.get(category, []):
                test_name = entry['test_name']
                if test_name not in pivoted:
                    pivoted[test_name] = {cat: [] for cat in CAT_KEYS}
                pivoted[test_name][category].append({**entry, 'job_name': job_name})

    return dict(sorted(pivoted.items()))


def _esc(text: str) -> str:
    return html_mod.escape(text)


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
.header{text-align:center;padding:20px 0 20px}
.header h1{font-size:1.8em;margin-bottom:10px;color:var(--fg-heading)}
.context{margin-bottom:8px;font-size:.9em;color:var(--fg-muted)}
.context a{color:var(--link)}
.context code{background:var(--bg-surface);
padding:2px 6px;border-radius:4px;font-size:.85em}
.subtitle{color:var(--fg-muted);font-size:.95em}
.nav{position:sticky;top:0;background:var(--bg);
border-bottom:1px solid var(--border-muted);
padding:10px 0;z-index:100;margin-bottom:20px}
.nav-row{display:flex;gap:8px;flex-wrap:wrap;
justify-content:center;align-items:center}
.nav-row+.nav-row{margin-top:6px}
.badge{padding:6px 14px;border-radius:20px;font-size:.85em;
font-weight:600;cursor:pointer;user-select:none;
transition:opacity .15s}
.badge.error{background:var(--err);color:#fff}
.badge.warning-error{background:var(--warn-err);color:#fff}
.badge.warning{background:var(--warn);color:#fff}
.badge.ok{background:var(--ok);color:#fff}
.badge.inactive{opacity:.3}
.search{padding:6px 12px;border-radius:8px;border:1px solid var(--border);
background:var(--bg-surface);color:var(--fg);font-size:.85em;
width:220px;outline:none}
.search:focus{border-color:var(--link)}
.matrix{width:100%;border-collapse:collapse;margin:0 auto 20px;
font-size:.8em;max-width:900px}
.matrix th,.matrix td{padding:4px 8px;border:1px solid var(--border);
text-align:center}
.matrix th{background:var(--bg-surface);color:var(--fg-heading);
font-weight:600;position:sticky;top:0}
.matrix td.job-label{text-align:left;font-weight:500;
max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.matrix td.count-err{color:var(--err-fg);font-weight:600}
.matrix td.count-warn-err{color:var(--warn-err);font-weight:600}
.matrix td.count-warn{color:var(--warn-fg);font-weight:600}
.matrix td.zero{color:var(--fg-muted);opacity:.4}
.matrix tr.job-row{cursor:pointer;transition:opacity .15s}
.matrix tr.job-row:hover{background:var(--nav-hover)}
.matrix tr.job-row.inactive{opacity:.3}
.test-group{margin-bottom:8px;border:1px solid var(--border);
border-radius:8px;background:var(--bg-surface);overflow:hidden}
.test-header{display:flex;align-items:center;gap:10px;
padding:10px 16px;cursor:pointer;user-select:none}
.test-header:hover{background:var(--nav-hover)}
.test-header h3{flex:1;font-size:.95em;color:var(--fg-heading);
word-break:break-all;margin:0}
.test-chips{display:flex;gap:4px;flex-wrap:wrap}
.chip{padding:2px 8px;border-radius:10px;font-size:.7em;font-weight:600}
.chip.error{background:var(--err);color:#fff}
.chip.warning-error{background:var(--warn-err);color:#fff}
.chip.warning{background:var(--warn);color:#fff}
.job-count{font-size:.8em;color:var(--fg-muted);white-space:nowrap}
.chevron{color:var(--fg-muted);font-size:.8em;transition:transform .15s}
.chevron.open{transform:rotate(90deg)}
.test-body{border-top:1px solid var(--border)}
.test-body.collapsed{display:none}
.job-section{padding:12px 16px;border-bottom:1px solid var(--border-muted)}
.job-section:last-child{border-bottom:none}
.job-label{font-size:.85em;font-weight:600;color:var(--fg-heading);
margin-bottom:8px}
.comparison{display:flex;gap:16px;flex-wrap:wrap}
.image-col{flex:1;min-width:250px}
.image-col h4{font-size:.8em;color:var(--fg-muted);margin-bottom:6px}
.cache-label{font-weight:normal}
.image-col img{width:100%;border:1px solid var(--border);
border-radius:4px;background:var(--bg-overlay)}
.no-image{padding:30px;text-align:center;
border:1px dashed var(--border);border-radius:4px;color:var(--fg-muted);
font-size:.85em}
.hidden{display:none}
.toggle-all{padding:4px 12px;border-radius:6px;border:1px solid var(--border);
background:var(--bg-surface);color:var(--fg);font-size:.8em;cursor:pointer}
.toggle-all:hover{background:var(--nav-hover)}
.empty-msg{text-align:center;padding:40px;color:var(--fg-muted);font-size:.95em}
"""

_JS = """\
<script>
document.addEventListener('DOMContentLoaded',()=>{
  const badges=document.querySelectorAll('.badge[data-cat]');
  const groups=document.querySelectorAll('.test-group');
  const search=document.getElementById('search');
  const jobRows=document.querySelectorAll('.matrix .job-row');
  const activeCats=new Set(['errors']);
  const activeJobs=new Set();

  function apply(){
    const q=(search?search.value:'').toLowerCase();
    const filterJobs=activeJobs.size>0;
    badges.forEach(b=>b.classList.toggle('inactive',!activeCats.has(b.dataset.cat)));
    jobRows.forEach(r=>r.classList.toggle('inactive',filterJobs&&!activeJobs.has(r.dataset.job)));
    let visible=0;
    groups.forEach(g=>{
      const nameMatch=!q||g.dataset.name.toLowerCase().includes(q);
      // Check if any visible job-section exists
      let hasVisible=false;
      g.querySelectorAll('.job-section').forEach(sec=>{
        const catOk=activeCats.has(sec.dataset.cat);
        const jobOk=!filterJobs||activeJobs.has(sec.dataset.job);
        const show=catOk&&jobOk;
        sec.classList.toggle('hidden',!show);
        if(show)hasVisible=true;
      });
      g.querySelectorAll('.chip').forEach(chip=>{
        chip.classList.toggle('hidden',!activeCats.has(chip.dataset.cat));
      });
      const show=hasVisible&&nameMatch;
      g.classList.toggle('hidden',!show);
      if(show)visible++;
    });
    const msg=document.getElementById('empty-msg');
    if(msg)msg.classList.toggle('hidden',visible>0);
  }

  badges.forEach(b=>b.addEventListener('click',()=>{
    const c=b.dataset.cat;
    if(activeCats.has(c))activeCats.delete(c);else activeCats.add(c);
    apply();
  }));

  jobRows.forEach(r=>r.addEventListener('click',()=>{
    const j=r.dataset.job;
    if(activeJobs.has(j))activeJobs.delete(j);else activeJobs.add(j);
    apply();
  }));

  if(search)search.addEventListener('input',apply);

  document.querySelectorAll('.test-header').forEach(h=>{
    h.addEventListener('click',()=>{
      const body=h.nextElementSibling;
      const chev=h.querySelector('.chevron');
      body.classList.toggle('collapsed');
      chev.classList.toggle('open');
    });
  });

  const toggleBtn=document.getElementById('toggle-all');
  if(toggleBtn){
    let allOpen=true;
    toggleBtn.addEventListener('click',()=>{
      allOpen=!allOpen;
      document.querySelectorAll('.test-body').forEach(b=>b.classList.toggle('collapsed',!allOpen));
      document.querySelectorAll('.chevron').forEach(c=>c.classList.toggle('open',allOpen));
      toggleBtn.textContent=allOpen?'Collapse All':'Expand All';
    });
  }

  // Start with chevrons open
  document.querySelectorAll('.chevron').forEach(c=>c.classList.add('open'));
  apply();
});
</script>"""


def _build_job_html(entry: dict[str, Any]) -> str:
    """Build one job's image comparison within a test group."""
    name = _esc(entry['test_name'])
    cache_cols = ''
    if entry['cache_rels']:
        for cr in entry['cache_rels']:
            label = _esc(cr['label'])
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
            '<div class="no-image">No baseline image</div>'
            '</div>'
        )

    return (
        f'<div class="comparison">{cache_cols}'
        f'<div class="image-col"><h4>Generated (Test)</h4>'
        f'<img src="{entry["test_rel"]}" alt="Generated: {name}" loading="lazy">'
        '</div></div>'
    )


def _build_matrix_html(
    results: dict[str, dict[str, list[dict[str, Any]]]],
) -> str:
    """Build a summary matrix: jobs as rows, categories as columns."""
    rows = ''
    for job_name, job_data in sorted(results.items()):
        cells = ''
        for cat, _, css_class in CATEGORIES:
            count = len(job_data.get(cat, []))
            if count:
                cells += f'<td class="count-{css_class.replace("-", "-")}">{count}</td>'
            else:
                cells += '<td class="zero">0</td>'
        safe = _esc(job_name)
        rows += (
            f'<tr class="job-row" data-job="{safe}">'
            f'<td class="job-label" title="{safe}">{safe}</td>{cells}</tr>'
        )

    return (
        '<table class="matrix"><thead><tr>'
        '<th>Job</th><th>Errors</th><th>Errors as Warnings</th><th>Warnings</th>'
        '</tr></thead><tbody>'
        f'{rows}'
        '</tbody></table>'
    )


def _build_html(
    results: dict[str, dict[str, list[dict[str, Any]]]],
    summary: dict[str, int],
    pr_url: str = '',
    branch: str = '',
) -> str:
    """Return the full HTML page as a string."""
    pivoted = _pivot_by_test(results)

    # Build test groups
    test_groups = ''
    for test_name, cat_data in pivoted.items():
        # Collect which categories this test appears in
        cats_present = [cat for cat in CAT_KEYS if cat_data.get(cat)]
        if not cats_present:
            continue

        # Chips showing category + count
        chips = ''
        for cat, label, css_class in CATEGORIES:
            entries = cat_data.get(cat, [])
            if entries:
                job_count = len(entries)
                chips += (
                    f'<span class="chip {css_class}" data-cat="{cat}">'
                    f'{job_count} {label.lower()}</span>'
                )

        total_jobs = sum(len(cat_data.get(cat, [])) for cat in CAT_KEYS)

        # Job sections inside the collapsible body
        job_sections = ''
        for cat, _, css_class in CATEGORIES:
            for entry in cat_data.get(cat, []):
                safe_job = _esc(entry['job_name'])
                job_sections += (
                    f'<div class="job-section" data-cat="{cat}" data-job="{safe_job}">'
                    f'<div class="job-label">{safe_job} '
                    f'<span class="chip {css_class}" style="font-size:.65em">{cat}</span></div>'
                    f'{_build_job_html(entry)}'
                    '</div>'
                )

        test_groups += (
            f'<div class="test-group" data-cats="{",".join(cats_present)}" '
            f'data-name="{_esc(test_name)}">'
            f'<div class="test-header">'
            f'<span class="chevron">&#9654;</span>'
            f'<h3>{_esc(test_name)}</h3>'
            f'<div class="test-chips">{chips}</div>'
            f'<span class="job-count">{total_jobs} job{"s" if total_jobs != 1 else ""}</span>'
            f'</div>'
            f'<div class="test-body">{job_sections}</div>'
            f'</div>'
        )

    # Filter badges
    badges = ''
    if summary['num_errors']:
        badges += (
            f'<span class="badge error" data-cat="errors">{summary["num_errors"]} errors</span>'
        )
    if summary['num_errors_as_warnings']:
        badges += (
            f'<span class="badge warning-error" data-cat="errors_as_warnings">'
            f'{summary["num_errors_as_warnings"]} errors as warnings</span>'
        )
    if summary['num_warnings']:
        badges += (
            f'<span class="badge warning" data-cat="warnings">'
            f'{summary["num_warnings"]} warnings</span>'
        )

    diff_s = 's' if summary['total_differences'] != 1 else ''
    job_s = 's' if summary['num_jobs'] != 1 else ''
    test_s = 's' if len(pivoted) != 1 else ''
    subtitle = (
        f'{len(pivoted)} unique test{test_s} with differences'
        f' across {summary["num_jobs"]} job{job_s}'
        f' ({summary["total_differences"]} total image difference{diff_s})'
    )

    context_parts = []
    if pr_url:
        context_parts.append(f'<a href="{_esc(pr_url)}">{_esc(pr_url)}</a>')
    if branch:
        context_parts.append(f'<code>{_esc(branch)}</code>')
    context_line = f'<p class="context">{" | ".join(context_parts)}</p>\n' if context_parts else ''

    matrix = _build_matrix_html(results)

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
        f'<p class="subtitle">{subtitle}</p>\n'
        '</div>\n'
        f'{matrix}'
        f'<nav class="nav">'
        f'<div class="nav-row">{badges} '
        f'<input type="text" class="search" id="search" '
        f'placeholder="Search test name..."> '
        f'<button class="toggle-all" id="toggle-all">Collapse All</button></div>'
        f'</nav>\n'
        f'{test_groups}\n'
        '<p id="empty-msg" class="empty-msg hidden">'
        'No tests match the current filters.</p>\n'
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

    for job_name, job_data in results.items():
        job_dir = images_dir / _safe_dirname(job_name)
        job_dir.mkdir(exist_ok=True)
        for category in CAT_KEYS:
            for entry in job_data.get(category, []):
                dest = job_dir / f'{category}_test_{entry["test_name"]}{entry["from_test"].suffix}'
                shutil.copy2(entry['from_test'], dest)
                entry['test_rel'] = dest.relative_to(output_dir).as_posix()

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
    (output_dir / 'summary.json').write_text(json.dumps(summary))

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
