#!/usr/bin/env python3
"""
Vault Context Builder — scans the Napkin Vault and produces a compact snapshot.

Output: /tmp/vault-context-snapshot.md
Contains: filename + first 3 lines + tags/status from frontmatter per file.
No LLM calls, no dependencies beyond stdlib.

Usage:
    python3 vault_context_builder.py
    # Or as input for local_worker tasks:
    # input: /tmp/vault-context-snapshot.md
"""

import os
import re
from datetime import datetime, timezone
from pathlib import Path

VAULT_ROOT = Path(os.environ.get("VAULT_DIR", "/Users/jens/.napkin"))
OUTPUT = Path("/tmp/vault-context-snapshot.md")

SCAN_DIRS = [
    "ax-concepts",
    "concepts",
    "ideas",
    "projects",
    "articles",
    "research",
    "skills",
]

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


def extract_frontmatter(text: str) -> dict:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}
    fm = {}
    for line in match.group(1).split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            fm[key.strip()] = val.strip()
    return fm


def scan_vault() -> list[dict]:
    entries = []
    for subdir in SCAN_DIRS:
        scan_path = VAULT_ROOT / subdir
        if not scan_path.exists():
            continue
        for md_file in sorted(scan_path.rglob("*.md")):
            if md_file.name.startswith("."):
                continue
            try:
                text = md_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            fm = extract_frontmatter(text)

            body_start = text.find("---", 3)
            if body_start > 0:
                body = text[body_start + 3:].strip()
            else:
                body = text.strip()

            first_lines = []
            for line in body.split("\n"):
                line = line.strip()
                if line and not line.startswith("---"):
                    first_lines.append(line)
                if len(first_lines) >= 3:
                    break

            rel_path = md_file.relative_to(VAULT_ROOT)
            entries.append({
                "path": str(rel_path),
                "status": fm.get("status", ""),
                "tags": fm.get("tags", ""),
                "type": fm.get("type", ""),
                "preview": "\n".join(first_lines),
            })
    return entries


def build_snapshot(entries: list[dict]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "---",
        f"generated: {ts}",
        f"files: {len(entries)}",
        f"vault: {VAULT_ROOT}",
        "---",
        "",
        "# Vault Context Snapshot",
        "",
    ]

    current_dir = ""
    for e in entries:
        dir_part = str(Path(e["path"]).parent)
        if dir_part != current_dir:
            current_dir = dir_part
            lines.append(f"\n## {current_dir}/\n")

        meta_parts = []
        if e["status"]:
            meta_parts.append(f"status:{e['status']}")
        if e["type"]:
            meta_parts.append(f"type:{e['type']}")
        if e["tags"]:
            meta_parts.append(f"tags:{e['tags']}")
        meta = f" ({', '.join(meta_parts)})" if meta_parts else ""

        lines.append(f"### {Path(e['path']).name}{meta}")
        if e["preview"]:
            lines.append(e["preview"])
        lines.append("")

    return "\n".join(lines)


def main():
    entries = scan_vault()
    snapshot = build_snapshot(entries)
    OUTPUT.write_text(snapshot, encoding="utf-8")
    print(f"Snapshot written: {OUTPUT} ({len(entries)} files, {len(snapshot)} chars)")


if __name__ == "__main__":
    main()
