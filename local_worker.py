#!/usr/bin/env python3
"""
Local Worker — Ollama-based task queue for local model delegation.

Watches a queue directory for YAML-frontmatter task files and routes them to local Ollama models.

Watches ~/.napkin/inbox/local-queue/ for YAML-frontmatter task files,
routes them to local Ollama models, writes outputs to caller-specified paths.

Task format (YAML frontmatter + optional body):

    ---
    task: research-note              # free-form label for logging
    model: gemma3:27b                # any Ollama model (gemma3:27b, qwen2.5:14b, llama3.1:8b, gemma4:e4b)
    input: /path/to/input.md         # optional — file content appended to prompt
    output: /path/to/output.md       # REQUIRED — where the result is written
    prompt: "short prompt"           # REQUIRED — instruction for the model
    timeout: 900                     # optional seconds, default 900
    system: "system prompt"          # optional — system prompt, sent separately
    temperature: 0.3                 # optional, default 0.3
    num_predict: 2000                # optional, default 2000 output tokens
    ---
    Additional prompt body (appended after the inline `prompt:`).
    This can be multi-line and use full markdown.

State machine:

    *.md               → pending (waiting to be picked up)
    *.md.processing    → being processed (rename-lock)
    *.md.done          → completed successfully
    *.md.failed        → failed (.failed file contains error detail)

Security:
- Only paths inside ALLOWED_PREFIXES are accepted for `input` and `output`.
- Model name must be on MODEL_ALLOWLIST.
- Task files that don't parse cleanly are renamed to *.malformed and skipped.

Location: ~/.membrane/local_worker.py
Launch:   ~/Library/LaunchAgents/com.membrane.local-worker.plist
Logs:     ~/.membrane/local-worker.log
"""

import atexit
import fcntl
import json
import os
import re
import shutil
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# --- Configuration ---

QUEUE_DIR = Path(os.environ.get("VAULT_DIR", os.path.expanduser("~/.napkin"))) / "inbox/local-queue"
LOG_FILE = Path(os.environ.get("MEMBRANE_DIR", os.path.expanduser("~/.membrane"))) / "local-worker.log"
LOCK_FILE = Path(os.environ.get("MEMBRANE_DIR", os.path.expanduser("~/.membrane"))) / "local-worker.lock"

OLLAMA_URL = "http://localhost:11434"
POLL_INTERVAL_S = 10

DEFAULT_TIMEOUT_S = 900
DEFAULT_TEMPERATURE = 0.3
DEFAULT_NUM_PREDICT = 2000

ALLOWED_PREFIXES = [
    Path(os.environ.get("VAULT_DIR", os.path.expanduser("~/.napkin"))),
    Path(os.environ.get("MEMBRANE_DIR", os.path.expanduser("~/.membrane"))),
    Path("/tmp"),
]

MODEL_ALLOWLIST = {
    "gemma3:27b",
    "qwen2.5:14b",
    "llama3.1:8b",
    "llama3.2:1b",
    "llama3.3:70b",
    "gemma4:e4b",
}


# --- Logging ---

def log(msg: str):
    """Append a timestamped line to the log file.
    Also echoes to stdout only if running interactively (TTY) — under launchd,
    stdout is piped to the same log file, which would cause duplicate entries."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{ts} {msg}"
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass  # don't crash worker on log failure
    if sys.stdout.isatty():
        print(line, flush=True)


# --- Single-instance lock ---

def acquire_singleton_lock():
    try:
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        fh = open(LOCK_FILE, "w")
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.write(f"{os.getpid()}\n{datetime.now(timezone.utc).isoformat()}\n")
        fh.flush()

        def _release():
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                fh.close()
                if LOCK_FILE.exists():
                    LOCK_FILE.unlink()
            except Exception:
                pass

        atexit.register(_release)
        return fh
    except (BlockingIOError, OSError):
        return None


# --- Task parsing ---

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n(.*))?$", re.DOTALL)


def parse_task(path: Path) -> dict | None:
    """Parse a task file. Returns dict or None if malformed."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        log(f"[parse] cannot read {path}: {e}")
        return None

    match = FRONTMATTER_RE.match(text)
    if not match:
        log(f"[parse] no frontmatter in {path.name}")
        return None

    frontmatter_text = match.group(1)
    body = match.group(2) or ""

    # Minimal YAML-ish parser: key: value, one per line
    task = {}
    for line in frontmatter_text.split("\n"):
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        # Strip quotes
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        # Numeric conversion
        if value and value.replace(".", "", 1).isdigit():
            value = float(value) if "." in value else int(value)
        task[key] = value

    # Body can extend the prompt
    if body.strip():
        prompt_body = body.strip()
        if "prompt" in task:
            task["prompt"] = f"{task['prompt']}\n\n{prompt_body}"
        else:
            task["prompt"] = prompt_body

    return task


# --- Validation ---

def path_in_allowed(p: Path) -> bool:
    try:
        abs_p = p.resolve()
    except Exception:
        return False
    for prefix in ALLOWED_PREFIXES:
        try:
            abs_p.relative_to(prefix.resolve())
            return True
        except ValueError:
            continue
    return False


def validate_task(task: dict) -> tuple[bool, str]:
    """Return (valid, error_message)."""
    if "model" not in task:
        return False, "missing 'model'"
    if task["model"] not in MODEL_ALLOWLIST:
        return False, f"model '{task['model']}' not in allowlist"
    if "output" not in task:
        return False, "missing 'output'"
    if not isinstance(task["output"], str):
        return False, "'output' must be a string path"
    if not path_in_allowed(Path(task["output"])):
        return False, f"output path '{task['output']}' not in allowed prefixes"
    if "input" in task and task["input"]:
        if not isinstance(task["input"], str):
            return False, "'input' must be a string path"
        if not path_in_allowed(Path(task["input"])):
            return False, f"input path '{task['input']}' not in allowed prefixes"
        if not Path(task["input"]).exists():
            return False, f"input path '{task['input']}' does not exist"
    if "prompt" not in task or not str(task["prompt"]).strip():
        return False, "missing or empty 'prompt'"
    return True, ""


# --- Ollama call ---

def call_ollama(task: dict) -> tuple[bool, str]:
    """Returns (success, output_text_or_error)."""
    prompt_parts = []
    if task.get("input"):
        try:
            input_content = Path(task["input"]).read_text(encoding="utf-8")
            prompt_parts.append("=== INPUT FILE ===")
            prompt_parts.append(input_content)
            prompt_parts.append("=== END INPUT ===\n")
        except Exception as e:
            return False, f"failed to read input file: {e}"

    prompt_parts.append(str(task["prompt"]))
    full_prompt = "\n".join(prompt_parts)

    payload = {
        "model": task["model"],
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": float(task.get("temperature", DEFAULT_TEMPERATURE)),
            "num_predict": int(task.get("num_predict", DEFAULT_NUM_PREDICT)),
        },
    }
    if task.get("system"):
        payload["system"] = str(task["system"])

    timeout_s = int(task.get("timeout", DEFAULT_TIMEOUT_S))

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=timeout_s,
        )
    except requests.exceptions.Timeout:
        return False, f"ollama timeout after {timeout_s}s"
    except Exception as e:
        return False, f"ollama request failed: {e}"

    if resp.status_code != 200:
        return False, f"ollama returned {resp.status_code}: {resp.text[:500]}"

    try:
        data = resp.json()
    except Exception as e:
        return False, f"ollama response not json: {e}"

    return True, data.get("response", "")


# --- Atomic output write ---

def write_output_atomic(output_path: Path, content: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=str(output_path.parent),
        suffix=".tmp",
        encoding="utf-8",
    )
    try:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
    finally:
        tmp.close()
    os.replace(tmp.name, output_path)
    # Force 644 on outputs regardless of process umask, so other users can
    # read the results the worker produced. Prevents the common issue where
    # launchd default umask 077 creates 600-mode files that no other user can see.
    try:
        os.chmod(output_path, 0o644)
    except Exception:
        pass


# --- Task processing ---

def process_task(task_file: Path):
    """Process one task file. Renames to .processing during, .done/.failed after."""
    processing = task_file.with_suffix(task_file.suffix + ".processing")
    try:
        task_file.rename(processing)
    except FileNotFoundError:
        return  # picked up by another poll tick

    started_at = time.time()
    task_label = "?"

    try:
        task = parse_task(processing)
        if task is None:
            processing.rename(task_file.with_suffix(task_file.suffix + ".malformed"))
            log(f"[task] malformed: {task_file.name}")
            return

        task_label = str(task.get("task", "?"))
        log(f"[task] start {task_file.name} task={task_label} model={task.get('model')}")

        ok, err = validate_task(task)
        if not ok:
            failed = task_file.with_suffix(task_file.suffix + ".failed")
            processing.rename(failed)
            with open(str(failed) + ".error", "w") as fh:
                fh.write(f"validation failed: {err}\n")
            log(f"[task] failed validation {task_file.name}: {err}")
            return

        success, result = call_ollama(task)
        if not success:
            failed = task_file.with_suffix(task_file.suffix + ".failed")
            processing.rename(failed)
            with open(str(failed) + ".error", "w") as fh:
                fh.write(f"ollama call failed: {result}\n")
            log(f"[task] failed ollama {task_file.name}: {result[:200]}")
            return

        try:
            write_output_atomic(Path(task["output"]), result)
        except Exception as e:
            failed = task_file.with_suffix(task_file.suffix + ".failed")
            processing.rename(failed)
            with open(str(failed) + ".error", "w") as fh:
                fh.write(f"output write failed: {e}\n")
            log(f"[task] failed write {task_file.name}: {e}")
            return

        done = task_file.with_suffix(task_file.suffix + ".done")
        processing.rename(done)
        duration = round(time.time() - started_at, 1)
        log(
            f"[task] done {task_file.name} task={task_label} "
            f"duration={duration}s output_chars={len(result)}"
        )
    except Exception as e:
        log(f"[task] unexpected error on {task_file.name}: {e}")
        try:
            if processing.exists():
                processing.rename(task_file.with_suffix(task_file.suffix + ".failed"))
        except Exception:
            pass


# --- Main loop ---

_shutdown_requested = False


def handle_signal(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    log(f"[worker] signal {signum} received, shutting down after current task")


def main():
    lock = acquire_singleton_lock()
    if lock is None:
        print("[worker] another instance is already running — exiting silently.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    log(f"[worker] starting (pid={os.getpid()}) queue={QUEUE_DIR}")
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)

    while not _shutdown_requested:
        try:
            # Only pick up files that have no extension suffix beyond .md
            pending = sorted(
                p for p in QUEUE_DIR.glob("*.md")
                if p.is_file() and not p.name.endswith((".processing", ".done", ".failed", ".malformed"))
            )
            for task_file in pending:
                if _shutdown_requested:
                    break
                process_task(task_file)
        except Exception as e:
            log(f"[worker] poll loop error: {e}")
        time.sleep(POLL_INTERVAL_S)

    log("[worker] shutdown complete")


if __name__ == "__main__":
    main()
