#!/usr/bin/env bash

# Usage: vtk-master-failure-summary.sh <out_dir> <tox log>...  (writes failure_summary.txt, n_failures.txt)
set -uo pipefail

out_dir="$1"
shift
mkdir -p "$out_dir"
summary="$out_dir/failure_summary.txt"
: >"$summary"
n_failures=0
for log in "$@"; do
	[ -s "$log" ] || continue
	# Strip ANSI codes, CR line endings and object addresses.
	clean="$(tr -d '\r' <"$log" | sed -E -e $'s/\x1b\\[[0-9;]*[a-zA-Z]//g' -e 's/0x[0-9a-fA-F]+/0x?/g')"
	start_line="$(printf '%s\n' "$clean" | grep -n -m1 -E '=+ short test summary info =+' | cut -d: -f1 || true)"
	if [ -n "$start_line" ]; then
		# FAILED/ERROR lines only, sorted: xdist order and SKIPPED counts vary per run.
		segment="$(printf '%s\n' "$clean" | tail -n "+${start_line}" | grep -E '^(FAILED|ERROR) ' | LC_ALL=C sort || true)"
		final="$(printf '%s\n' "$clean" | grep -E '^=+ .* in [0-9.]+s' | tail -n1 || true)"
		for n in $(printf '%s' "$final" | grep -oE '[0-9]+ (failed|errors?)' | cut -d' ' -f1); do
			n_failures=$((n_failures + n))
		done
	else
		segment="$(printf '%s\n' "$clean" | tail -n 40)"
	fi
	[ -n "$segment" ] || continue
	# Cap each line before the size cap: a single repr can run to kilobytes.
	printf '%s\n' "$segment" | sed -E 's/^(.{200}).+$/\1.../' >>"$summary"
done
# Three jobs' summaries must fit GitHub's 65536-character comment.
if [ "$(wc -c <"$summary")" -gt 16000 ]; then
	head -c 16000 "$summary" | sed '$d' >"$summary.tmp"
	mv "$summary.tmp" "$summary"
fi
printf '%s\n' "$n_failures" >"$out_dir/n_failures.txt"
