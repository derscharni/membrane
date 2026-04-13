#!/usr/bin/env python3
"""
Context Compressor — Dependency-Free Context Compression for Agent Sessions.

Reduces token consumption in long sessions without external dependencies.
Pure Python, single file, fully auditable. No ML models, no external APIs,
no semantic similarity. Just deterministic rules.

Three-stage context compression:
  Stufe 1: Tool-Output-Kompression
  Stufe 2: Message-History-Pruning
  Stufe 3: Rolling Window

Location: ~/.membrane/context_compressor.py
Log:      ~/.membrane/compressor.log

Usage:
    from context_compressor import compress_messages
    compressed, stats = compress_messages(messages, config={...})
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG_FILE = Path(os.environ.get("MEMBRANE_DIR", os.path.expanduser("~/.membrane"))) / "compressor.log"

# Framework terms that mark "important" assistant messages (never pruned in Stufe 2)
FRAMEWORK_TERMS = {
    "ax stack", "ax-stack", "context sovereignty", "context-sovereignty",
    "killerjourney", "killer journey", "temporal ux", "temporal-ux",
    "temporal integrity", "harness engineering", "harness-engineering",
    "context handshake", "context-handshake", "trust stack", "trust-stack",
    "authority layer", "accountability", "agent experience",
    "relationship layer", "orchestration", "archivar", "confidence scoring",
    "supersession", "sycophancy", "sovereignty",
}

# Short ACK messages that can be safely pruned (Stufe 2)
ACK_PATTERNS = re.compile(
    r"^(ok|okay|gut|ja|jo|mach|los|go|done|fertig|passt|genau|weiter|next|"
    r"verstanden|alles klar|check|yep|yes|sure|thx|danke|👍|✅|n8|gi)\.?!?$",
    re.IGNORECASE,
)

DEFAULT_CONFIG = {
    "tool_output_max": 2000,
    "rolling_window": 20,
    "keep_recent": 10,
    "truncate_assistant_after": 500,
    "truncate_to": 200,
    "min_user_content_keep": 100,
}


# ─── Stufe 1: Tool-Output-Kompression ────────────────────────────────────────

def compress_tool_output(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content

    # JSON detection: try to parse and summarize
    stripped = content.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            obj = json.loads(stripped)
            return _compress_json(obj, max_chars)
        except (json.JSONDecodeError, ValueError):
            pass

    # File-read pattern: keep first 50 + last 20 lines
    lines = content.split("\n")
    if len(lines) > 80:
        head = lines[:50]
        tail = lines[-20:]
        removed = len(lines) - 70
        return "\n".join(head) + f"\n\n[...{removed} lines truncated...]\n\n" + "\n".join(tail)

    # Search-result pattern: if lines look like "Title\nURL\ndescription..."
    if _looks_like_search_results(lines):
        return _compress_search_results(lines, max_chars)

    # Repetitive structure detection
    compressed = _compress_repetitive(content, max_chars)
    if compressed:
        return compressed

    # Fallback: hard truncate with marker
    return content[:max_chars] + f"\n[...truncated, {len(content) - max_chars} chars removed...]"


def _compress_json(obj, max_chars: int) -> str:
    if isinstance(obj, dict):
        keys = list(obj.keys())
        if len(keys) <= 5:
            return json.dumps(obj, indent=2, ensure_ascii=False)[:max_chars]
        summary_keys = keys[:3] + keys[-2:]
        summary = {k: obj[k] for k in summary_keys if k in obj}
        truncated = len(keys) - len(summary_keys)
        result = json.dumps(summary, indent=2, ensure_ascii=False)
        return result + f"\n// ...{truncated} fields truncated"

    if isinstance(obj, list):
        if len(obj) <= 3:
            return json.dumps(obj, indent=2, ensure_ascii=False)[:max_chars]
        sample = obj[:2] + [f"...({len(obj) - 3} more items)..."] + obj[-1:]
        return json.dumps(sample, indent=2, ensure_ascii=False)

    return json.dumps(obj, ensure_ascii=False)[:max_chars]


def _looks_like_search_results(lines: list) -> bool:
    url_count = sum(1 for l in lines if re.match(r"https?://", l.strip()))
    return url_count >= 3 and url_count > len(lines) * 0.1


def _compress_search_results(lines: list, max_chars: int) -> str:
    results = []
    current = []
    for line in lines:
        if re.match(r"https?://", line.strip()) and current:
            results.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        results.append(current)

    compressed_parts = []
    for result_lines in results[:10]:
        title = result_lines[0] if result_lines else ""
        url = ""
        snippet = ""
        for rl in result_lines:
            if re.match(r"https?://", rl.strip()):
                url = rl.strip()
            elif len(rl.strip()) > 20 and not url:
                snippet = rl.strip()[:200]
        compressed_parts.append(f"{title}\n{url}\n{snippet[:200]}")

    truncated = max(0, len(results) - 10)
    result = "\n\n".join(compressed_parts)
    if truncated:
        result += f"\n\n[...{truncated} more results truncated...]"
    return result[:max_chars]


def _compress_repetitive(content: str, max_chars: int) -> str | None:
    lines = content.split("\n")
    if len(lines) < 6:
        return None

    # Detect structural repetition: lines with similar prefix patterns
    prefixes = {}
    for line in lines:
        prefix = line[:30] if len(line) > 30 else line
        prefix_key = re.sub(r"\d+", "N", prefix)
        prefixes[prefix_key] = prefixes.get(prefix_key, 0) + 1

    repeated = {k: v for k, v in prefixes.items() if v >= 3}
    if not repeated:
        return None

    most_common = max(repeated, key=repeated.get)
    count = repeated[most_common]

    # Keep first 3, summarize rest
    kept = []
    skipped = 0
    for line in lines:
        key = re.sub(r"\d+", "N", line[:30] if len(line) > 30 else line)
        if key == most_common and skipped < count - 3:
            if len(kept) < 3 or key != most_common:
                kept.append(line)
            else:
                skipped += 1
        else:
            kept.append(line)

    if skipped > 0:
        kept.append(f"[...{skipped} similar entries omitted...]")

    result = "\n".join(kept)
    return result if len(result) < len(content) else None


# ─── Stufe 2: Message-History-Pruning ─────────────────────────────────────────

def _get_message_text(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    parts.append(str(block.get("content", "")))
        return "\n".join(parts)
    return str(content)


def _is_ack_message(msg: dict) -> bool:
    text = _get_message_text(msg).strip()
    return len(text) < 20 and bool(ACK_PATTERNS.match(text))


def _has_framework_terms(msg: dict) -> bool:
    text = _get_message_text(msg).lower()
    return any(term in text for term in FRAMEWORK_TERMS)


def _is_duplicate_tool_call(msg: dict, seen_calls: set) -> bool:
    if msg.get("role") != "assistant":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            sig = f"{block.get('name', '')}:{json.dumps(block.get('input', {}), sort_keys=True)}"
            if sig in seen_calls:
                return True
            seen_calls.add(sig)
    return False


def _is_empty_tool_result(msg: dict) -> bool:
    if msg.get("role") != "tool":
        return False
    text = _get_message_text(msg).strip()
    return len(text) == 0


def should_prune(msg: dict, index: int, total: int, config: dict, seen_calls: set) -> bool:
    keep_recent = config.get("keep_recent", DEFAULT_CONFIG["keep_recent"])
    min_user = config.get("min_user_content_keep", DEFAULT_CONFIG["min_user_content_keep"])

    # Never prune recent messages
    if index >= total - keep_recent:
        return False

    # Never prune system messages
    if msg.get("role") == "system":
        return False

    # Never prune user messages with substantial content
    if msg.get("role") == "user" and len(_get_message_text(msg)) > min_user:
        return False

    # Never prune assistant messages with framework terms
    if msg.get("role") == "assistant" and _has_framework_terms(msg):
        return False

    # Prune ACK messages
    if _is_ack_message(msg):
        return True

    # Prune empty tool results
    if _is_empty_tool_result(msg):
        return True

    # Prune duplicate tool calls
    if _is_duplicate_tool_call(msg, seen_calls):
        return True

    return False


# ─── Stufe 3: Rolling Window ─────────────────────────────────────────────────

def apply_rolling_window(messages: list, config: dict) -> list:
    rolling_window = config.get("rolling_window", DEFAULT_CONFIG["rolling_window"])
    truncate_after = config.get("truncate_assistant_after", DEFAULT_CONFIG["truncate_assistant_after"])
    truncate_to = config.get("truncate_to", DEFAULT_CONFIG["truncate_to"])

    if len(messages) <= rolling_window:
        return messages

    result = []
    boundary = len(messages) - rolling_window

    for i, msg in enumerate(messages):
        if i >= boundary:
            result.append(msg)
            continue

        role = msg.get("role", "")

        # System messages pass through
        if role == "system":
            result.append(msg)
            continue

        # Assistant messages: truncate long ones
        if role == "assistant":
            text = _get_message_text(msg)
            if len(text) > truncate_after:
                truncated_msg = dict(msg)
                truncated_text = text[:truncate_to] + f"\n[...komprimiert, {len(text) - truncate_to} Zeichen entfernt]"
                truncated_msg["content"] = truncated_text
                result.append(truncated_msg)
            else:
                result.append(msg)
            continue

        # Tool results: keep only summary line
        if role == "tool":
            text = _get_message_text(msg)
            if len(text) > truncate_after:
                lines = text.split("\n")
                summary = lines[0] if lines else ""
                truncated_msg = dict(msg)
                truncated_msg["content"] = summary + f"\n[...tool output komprimiert, {len(text)} chars → {len(summary)} chars]"
                result.append(truncated_msg)
            else:
                result.append(msg)
            continue

        # User messages outside rolling window: keep as-is (already filtered by Stufe 2)
        result.append(msg)

    return result


# ─── Main Interface ──────────────────────────────────────────────────────────

def compress_messages(messages: list, config: dict | None = None) -> tuple[list, dict]:
    """
    Compress a message array for reduced token consumption.

    Args:
        messages: List of message dicts (OpenAI/Anthropic format)
                  Each: {"role": "user"|"assistant"|"system"|"tool", "content": ...}
        config: Optional overrides for compression parameters.

    Returns:
        (compressed_messages, stats)
        stats: {"original_count", "compressed_count", "chars_before", "chars_after",
                "chars_removed", "ratio", "pruned_count", "truncated_count"}
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    original_count = len(messages)
    chars_before = sum(len(_get_message_text(m)) for m in messages)

    # --- Stufe 1: Compress tool outputs ---
    tool_max = cfg["tool_output_max"]
    stufe1 = []
    for msg in messages:
        if msg.get("role") == "tool" or (
            msg.get("role") == "assistant"
            and isinstance(msg.get("content"), list)
            and any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in msg.get("content", [])
            )
        ):
            text = _get_message_text(msg)
            if len(text) > tool_max:
                compressed_msg = dict(msg)
                compressed_msg["content"] = compress_tool_output(text, tool_max)
                stufe1.append(compressed_msg)
                continue
        stufe1.append(msg)

    # --- Stufe 2: Prune unnecessary messages ---
    seen_calls = set()
    stufe2 = []
    pruned_count = 0
    for i, msg in enumerate(stufe1):
        if should_prune(msg, i, len(stufe1), cfg, seen_calls):
            pruned_count += 1
        else:
            stufe2.append(msg)

    # --- Stufe 3: Rolling window truncation ---
    stufe3 = apply_rolling_window(stufe2, cfg)

    # --- Stats ---
    chars_after = sum(len(_get_message_text(m)) for m in stufe3)
    chars_removed = chars_before - chars_after
    ratio = round(chars_removed / chars_before, 3) if chars_before > 0 else 0.0

    stats = {
        "original_count": original_count,
        "compressed_count": len(stufe3),
        "chars_before": chars_before,
        "chars_after": chars_after,
        "chars_removed": chars_removed,
        "ratio": ratio,
        "pruned_count": pruned_count,
        "truncated_count": original_count - len(stufe2),
    }

    _log_compression(stats)
    return stufe3, stats


def _log_compression(stats: dict):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = (
        f"{ts} | before={stats['chars_before']} after={stats['chars_after']} "
        f"ratio={stats['ratio']:.1%} msgs={stats['original_count']}→{stats['compressed_count']} "
        f"pruned={stats['pruned_count']}"
    )
    try:
        with open(LOG_FILE, "a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


# ─── CLI for testing ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 context_compressor.py <session.jsonl>")
        print("  Reads a JSONL file of messages, compresses, and prints stats.")
        print("  Each line: {\"role\": \"...\", \"content\": \"...\"}")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    messages = []
    for line in path.read_text().strip().split("\n"):
        if line.strip():
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(messages)} messages from {path}")
    compressed, stats = compress_messages(messages)

    print(f"\n--- Compression Stats ---")
    print(f"Messages:  {stats['original_count']} → {stats['compressed_count']}")
    print(f"Chars:     {stats['chars_before']:,} → {stats['chars_after']:,}")
    print(f"Removed:   {stats['chars_removed']:,} ({stats['ratio']:.1%})")
    print(f"Pruned:    {stats['pruned_count']} messages removed")

    if "--output" in sys.argv:
        out_path = Path(sys.argv[sys.argv.index("--output") + 1])
        with open(out_path, "w") as fh:
            for msg in compressed:
                fh.write(json.dumps(msg, ensure_ascii=False) + "\n")
        print(f"\nCompressed output written to {out_path}")
