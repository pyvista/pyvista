#!/usr/bin/env bash

# Failure report helpers: source them, or run `hash < summary` / `label <n_failures> < summary`.

# Strip timings, object addresses and line order.
normalize_for_hash() {
	grep -Ev '^\s*(py3\.[0-9]+-[a-z_-]+: (exit|FAIL code)|evaluation failed|congratulations|py3\.[0-9]+-[a-z_-]+: OK)' |
		sed -E -e 's/ in [0-9]+\.[0-9]+s \([0-9:]+\)//' -e 's/0x[0-9a-fA-F]+/0x?/g' |
		LC_ALL=C sort
}

failure_hash() {
	normalize_for_hash | sha256sum | cut -d' ' -f1
}

# Summary label for one job from pytest's count, or the summary's lines when unknown.
summary_label() {
	local count="$1" summary="$2" n_lines n_failures label
	n_lines="$(printf '%s\n' "$summary" | grep -cE '^(FAILED|ERROR) ' || true)"
	n_failures="$count"
	[ "$n_failures" -gt 0 ] 2>/dev/null || n_failures="$n_lines"
	if [ "$n_failures" -eq 0 ]; then
		label="failed before pytest reported (end of log)"
	elif [ "$n_failures" -eq 1 ]; then
		label="1 failure"
	else
		label="${n_failures} failures"
	fi
	if [ "$n_lines" -gt 0 ] && [ "$n_lines" -lt "$n_failures" ]; then
		label+=", first ${n_lines} shown"
	fi
	printf '%s\n' "$label"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
	set -uo pipefail
	case "$1" in
	hash) failure_hash ;;
	label) summary_label "$2" "$(cat)" ;;
	*)
		echo "usage: $0 hash|label <n_failures>" >&2
		exit 2
		;;
	esac
fi
