#!/usr/bin/env python3
"""Look up which test-area YAML file (and priority) a Buildkite label belongs to.

Usage:
    ./lookup_test_area.py "Basic Correctness"
    ./lookup_test_area.py "Basic Correctness" "Cudagraph" "Nonexistent Label"
"""

import re
import sys
from pathlib import Path

# File (without .yaml) -> priority. Only these files are searched.
PRIORITY = {
    "e2e_integration": 1,
    "basic_correctness": 2,
    "lora": 2,
    "ray_compat": 2,
    "cuda": 3,
    "lm_eval": 3,
    "model_runner_v2": 3,
    "models_multimodal": 3,
}

TEST_AREAS_DIR = Path(__file__).resolve().parent / ".buildkite" / "test_areas"

# Matches an active (non-commented) step label line, e.g.:
#   - label: Basic Correctness
#   - label: "Multi-Modal Models (Standard) 1: qwen2"
LABEL_RE = re.compile(r"^\s*-\s*label:\s*(.+?)\s*$")


def clean_label(raw: str) -> str:
    """Strip surrounding quotes and any trailing inline `# ...` comment."""
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] in "\"'" and raw[-1] == raw[0]:
        return raw[1:-1]
    # Remove trailing inline comment (" # ...") that isn't inside quotes.
    raw = re.sub(r"\s+#.*$", "", raw)
    return raw.strip()


def build_index() -> dict[str, tuple[str, int]]:
    """Map each label -> (yaml_filename, priority)."""
    index: dict[str, tuple[str, int]] = {}
    for name, priority in PRIORITY.items():
        path = TEST_AREAS_DIR / f"{name}.yaml"
        if not path.exists():
            print(f"warning: {path} not found", file=sys.stderr)
            continue
        for line in path.read_text().splitlines():
            if line.lstrip().startswith("#"):
                continue  # commented-out step
            m = LABEL_RE.match(line)
            if m:
                index[clean_label(m.group(1))] = (f"{name}.yaml", priority)
    return index


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: lookup_test_area.py <label> [<label> ...]", file=sys.stderr)
        return 1

    index = build_index()
    for query in argv:
        exact = index.get(query)
        if exact:
            filename, priority = exact
            print(f"{query}: {filename} (priority {priority})")
            continue

        partial = [
            (label, filename, priority)
            for label, (filename, priority) in index.items()
            if query in label
        ]
        if not partial:
            print(f"{query}: N/A")
        elif len(partial) == 1:
            label, filename, priority = partial[0]
            print(f"{query}: {filename} (priority {priority}) [matched: {label}]")
        else:
            print(f"{query}: {len(partial)} matches")
            for label, filename, priority in partial:
                print(f"  - {label}: {filename} (priority {priority})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
