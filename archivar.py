#!/usr/bin/env python3
"""
Archivar — Vault-Pflege mit Sycophancy-Check und Feedback-Loop.

Läuft nachts (Cron 03:00). Liest Vault + SESSION.md, bewertet Ideen,
schlägt Status-Änderungen und Verbindungen vor. Alles als Proposal —
no direct write-back without explicit approval.

Sycophancy-Check: Jede Bewertung wird gegen Bias geprüft.
Feedback-Loop: reviewer evaluates proposals, Archivar learns from feedback.

Location: ~/.membrane/archivar.py
Config: model_router.py (task: archival → gemma3:27b)
Cron: 0 3 * * * cd ~/.membrane && python3 archivar.py >> archivar.log 2>&1
"""

import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model_router import ModelRouter, generate_local
from sanitizer import sanitize
from sycophancy import check_evaluation

# --- Paths ---
VAULT_DIR = Path(os.environ.get("VAULT_DIR", os.path.expanduser("~/.napkin")))
IDEAS_DIR = VAULT_DIR / "ideas"
BRIEFINGS_DIR = VAULT_DIR / "research/daily-briefings"
SESSION_FILE = VAULT_DIR / "SESSION.md"
INBOX_DIR = VAULT_DIR / "inbox/claude"
MEMBRANE_DIR = Path(os.environ.get("MEMBRANE_DIR", os.path.expanduser("~/.membrane")))
FEEDBACK_FILE = VAULT_DIR / "inbox" / "archivar-feedback.jsonl"
AUDIT_LOG = MEMBRANE_DIR / "archivar.log"
SYCOPHANCY_LOG = MEMBRANE_DIR / "sycophancy-log.jsonl"

# --- Confidence Scoring ---
STATUS_CONFIDENCE = {
    "seed": 0.2,
    "growing": 0.5,
    "ripe": 0.8,
    "stale": 0.1,
    "published": 1.0,
}
DECAY_AGE_DAYS = 14
DECAY_PER_WEEK = 0.05
EXTERNAL_CONFIRM_BOOST = 0.1
APPROVED_BOOST = 0.2
REJECTED_PENALTY = 0.3
STALE_THRESHOLD = 0.2
SUPERSESSION_MIN_SHARED_TAGS = 3

# --- Router ---
_router = None

def get_router():
    global _router
    if _router is None:
        _router = ModelRouter()
        _router.health.update()
    return _router


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{ts} {msg}"
    print(line)
    with open(AUDIT_LOG, "a") as f:
        f.write(line + "\n")


# --- Vault Reading ---

def read_ideas():
    """Read all idea files with their frontmatter status, tags, last-confirmed, existing confidence."""
    ideas = []
    if not IDEAS_DIR.exists():
        return ideas
    for md in IDEAS_DIR.glob("*.md"):
        if md.name.lower() == "readme.md":
            continue
        text = md.read_text(encoding="utf-8")
        text, _ = sanitize(text)
        head = text[:800]

        status = "unknown"
        match = re.search(r"^status:\s*(\w+)", head, re.MULTILINE)
        if match:
            status = match.group(1)

        tags = []
        tags_match = re.search(r"^tags:\s*\[([^\]]*)\]", head, re.MULTILINE)
        if tags_match:
            tags = [t.strip() for t in tags_match.group(1).split(",") if t.strip()]

        last_confirmed = None
        lc_match = re.search(r"^last-confirmed:\s*(\d{4}-\d{2}-\d{2})", head, re.MULTILINE)
        if lc_match:
            try:
                last_confirmed = datetime.strptime(lc_match.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        existing_confidence = None
        cf_match = re.search(r"^confidence:\s*([\d.]+)", head, re.MULTILINE)
        if cf_match:
            try:
                existing_confidence = float(cf_match.group(1))
            except ValueError:
                pass

        ideas.append({
            "path": str(md),
            "name": md.stem,
            "status": status,
            "tags": tags,
            "last_confirmed": last_confirmed,
            "existing_confidence": existing_confidence,
            "content": text[:1000],
            "modified": datetime.fromtimestamp(md.stat().st_mtime, tz=timezone.utc),
        })
    return ideas


def read_recent_briefings(days=7):
    """Read briefings from the last N days."""
    briefings = []
    if not BRIEFINGS_DIR.exists():
        return briefings
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    for md in sorted(BRIEFINGS_DIR.glob("*.md"), reverse=True):
        mtime = datetime.fromtimestamp(md.stat().st_mtime, tz=timezone.utc)
        if mtime < cutoff:
            break
        text = md.read_text(encoding="utf-8")
        text, _ = sanitize(text)
        briefings.append({"name": md.stem, "content": text[:2000]})
    return briefings


def read_session():
    """Read SESSION.md for recent context."""
    if not SESSION_FILE.exists():
        return ""
    text = SESSION_FILE.read_text(encoding="utf-8")
    text, _ = sanitize(text)
    return text[:3000]


def read_feedback():
    """Read previous feedback entries."""
    feedback = []
    if not FEEDBACK_FILE.exists():
        return feedback
    try:
        raw = FEEDBACK_FILE.read_text()
    except (PermissionError, OSError) as e:
        log(f"[FEEDBACK] Cannot read {FEEDBACK_FILE}: {e}")
        return feedback
    for line in raw.strip().split("\n"):
        if line.strip():
            try:
                feedback.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return feedback[-20:]  # Last 20 feedback entries


# --- Evaluation ---

def evaluate_idea(idea, briefings_summary, session_summary, feedback_summary):
    """Evaluate a single idea for resonance, external confirmation, and connections."""
    router = get_router()
    model = router.route("archival")
    if not model:
        log("[ERROR] No model available for archival")
        return None

    prompt = f"""You are an Archivar — a careful evaluator of ideas in a knowledge vault.
Your job: assess whether this idea is gaining traction, stalling, or fading.

IDEA:
Name: {idea['name']}
Current status: {idea['status']}
Last modified: {idea['modified'].strftime('%Y-%m-%d')}
Content preview: {idea['content'][:500]}

RECENT BRIEFINGS (last 7 days, summarized):
{briefings_summary[:1000]}

RECENT SESSION CONTEXT:
{session_summary[:500]}

FEEDBACK FROM PREVIOUS EVALUATIONS:
{feedback_summary}

Evaluate:
1. RESONANCE: How often does this topic appear in recent briefings/sessions? (0-3)
2. EXTERNAL: Are there new sources supporting or challenging this idea? (0-3)
3. CONNECTIONS: Are there obvious links to other ideas not yet made? List them.
4. STATUS RECOMMENDATION: Should status change? (keep / seed→growing / growing→ripe / growing→stale / ripe→published / no change)
5. REASONING: One paragraph explaining your assessment.

IMPORTANT: Be honest. If an idea is losing relevance, say so. Do not inflate scores to please the author.

Respond as JSON:
{{"resonance": 0, "external": 0, "connections": [], "status_recommendation": "no change", "reasoning": "..."}}"""

    result_text = generate_local(model, prompt, options={"temperature": 0.3, "num_predict": 800})
    if not result_text:
        return None

    cleaned = result_text.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            log(f"[ERROR] JSON parse failed for idea {idea['name']}")
    return None


def sycophancy_check(idea, evaluation):
    """Check if the evaluation shows sycophantic bias via shared module."""
    result, was_biased = check_evaluation(idea["name"], idea["status"], evaluation)
    if was_biased:
        log(f"[SYCOPHANCY] {idea['name']}: {result.get('sycophancy_flag', 'flagged')}")
    return result, was_biased


# --- Confidence Scoring + Supersession + Decay ---

def compute_confidence(idea, evaluation, feedback_history):
    """Compute confidence (0.0-1.0) from status + evaluation + feedback history + decay."""
    base = STATUS_CONFIDENCE.get(idea["status"], 0.3)
    confidence = base

    if evaluation and evaluation.get("external", 0) >= 2:
        confidence += EXTERNAL_CONFIRM_BOOST

    positive_verdicts = {"ALLOW", "APPROVE", "APPROVED", "OK", "YES", "ACCEPT"}
    negative_verdicts = {"DENY", "REJECT", "REJECTED", "NO", "BLOCK"}
    for fb in feedback_history:
        if fb.get("idea") != idea["name"]:
            continue
        verdict = str(fb.get("verdict", "")).strip().upper()
        if verdict in positive_verdicts:
            confidence += APPROVED_BOOST
        elif verdict in negative_verdicts:
            confidence -= REJECTED_PENALTY

    if idea.get("last_confirmed"):
        age_days = (datetime.now(timezone.utc) - idea["last_confirmed"]).days
        weeks_stale = max(0, (age_days - DECAY_AGE_DAYS) // 7)
        confidence -= weeks_stale * DECAY_PER_WEEK

    return max(0.0, min(1.0, round(confidence, 2)))


def detect_supersessions(ideas):
    """Detect possible supersessions: ideas sharing ≥2 tags, newer could supersede older."""
    proposals = []
    candidates = [i for i in ideas if i["status"] not in ("stale", "published") and i["tags"]]
    candidates.sort(key=lambda i: i["modified"])

    for idx, older in enumerate(candidates):
        older_tags = set(older["tags"])
        for newer in candidates[idx + 1:]:
            if newer["name"] == older["name"]:
                continue
            shared = older_tags & set(newer["tags"])
            if len(shared) >= SUPERSESSION_MIN_SHARED_TAGS:
                proposals.append({
                    "older": older["name"],
                    "newer": newer["name"],
                    "older_status": older["status"],
                    "shared_tags": sorted(shared),
                })
    return proposals


def apply_decay_check(idea):
    """Return True if idea has decayed below STALE_THRESHOLD and should be proposed stale."""
    if not idea.get("last_confirmed"):
        return False
    if idea["status"] in ("seed", "stale", "published"):
        return False
    age_days = (datetime.now(timezone.utc) - idea["last_confirmed"]).days
    if age_days <= DECAY_AGE_DAYS:
        return False
    base = idea.get("existing_confidence")
    if base is None:
        base = STATUS_CONFIDENCE.get(idea["status"], 0.3)
    weeks_stale = max(0, (age_days - DECAY_AGE_DAYS) // 7)
    projected = base - weeks_stale * DECAY_PER_WEEK
    return projected < STALE_THRESHOLD


_FRONTMATTER_RE = re.compile(r"^(---\s*\n)(.*?)(\n---\s*(?:\n|$))", re.DOTALL)

def update_idea_frontmatter(idea_path, confidence, last_confirmed_date):
    """Write confidence + last-confirmed into idea frontmatter. Preserves all other fields."""
    path = Path(idea_path)
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return False

    start_marker, fm_body, end_marker = match.group(1), match.group(2), match.group(3)
    body_after = text[match.end():]

    if re.search(r"^confidence:\s*", fm_body, re.MULTILINE):
        fm_body = re.sub(r"^confidence:\s*.*$", f"confidence: {confidence}", fm_body, flags=re.MULTILINE)
    else:
        fm_body = fm_body.rstrip() + f"\nconfidence: {confidence}"

    if re.search(r"^last-confirmed:\s*", fm_body, re.MULTILINE):
        fm_body = re.sub(r"^last-confirmed:\s*.*$", f"last-confirmed: {last_confirmed_date}", fm_body, flags=re.MULTILINE)
    else:
        fm_body = fm_body.rstrip() + f"\nlast-confirmed: {last_confirmed_date}"

    new_text = start_marker + fm_body + end_marker + body_after
    path.write_text(new_text, encoding="utf-8")
    return True


# --- Proposal Generation ---

def generate_proposals(evaluations, supersession_proposals=None, decay_proposals=None):
    """Write proposals to inbox for review."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    proposals_file = INBOX_DIR / f"{today}-archivar-proposals.md"
    supersession_proposals = supersession_proposals or []
    decay_proposals = decay_proposals or []

    lines = [f"# Archivar Proposals — {today}\n"]
    lines.append("*Automatically generated. Please review before applying.*\n")
    lines.append("*Feedback via: ALLOW / DENY / ADJUST pro Eintrag → wird in nächsten Run einbezogen.*\n")

    status_changes = []
    connections = []
    sycophancy_flags = []

    for idea_name, eval_data in evaluations.items():
        if not eval_data:
            continue

        rec = eval_data.get("status_recommendation", "no change")
        if rec != "no change":
            flag = " ⚠️ BIAS-ADJUSTED" if eval_data.get("sycophancy_flag") else ""
            conf = eval_data.get("confidence")
            conf_str = f" · confidence: {conf}" if conf is not None else ""
            status_changes.append(
                f"- **{idea_name}**: {rec} "
                f"(R:{eval_data.get('resonance', 0)}/3 E:{eval_data.get('external', 0)}/3){conf_str}{flag}\n"
                f"  Grund: {eval_data.get('reasoning', 'n/a')[:120]}"
            )

        conns = eval_data.get("connections", [])
        if conns:
            connections.append(f"- **{idea_name}** → {', '.join(conns[:5])}")

        if eval_data.get("sycophancy_flag"):
            sycophancy_flags.append(
                f"- **{idea_name}**: {eval_data['sycophancy_flag']} — {eval_data.get('sycophancy_note', '')}"
            )

    if status_changes:
        lines.append("\n## Status-Änderungen vorgeschlagen\n")
        lines.extend(status_changes)

    if connections:
        lines.append("\n## Neue Verbindungen vorgeschlagen\n")
        lines.extend(connections)

    if supersession_proposals:
        lines.append("\n## Supersession vorgeschlagen\n")
        lines.append("*String-matching auf Tags (≥2 gemeinsam) + Recency. Kein Auto-Write.*\n")
        for sp in supersession_proposals:
            lines.append(
                f"- **{sp['older']}** könnte superseded by **{sp['newer']}** sein\n"
                f"  Grund: geteilte Tags [{', '.join(sp['shared_tags'])}], {sp['newer']} ist neuer, "
                f"{sp['older']} hat status {sp['older_status']}"
            )

    if decay_proposals:
        lines.append("\n## Decay — propose stale\n")
        lines.append(f"*last-confirmed > {DECAY_AGE_DAYS} Tage und projected confidence < {STALE_THRESHOLD}.*\n")
        for idea in decay_proposals:
            lc = idea["last_confirmed"].strftime("%Y-%m-%d") if idea.get("last_confirmed") else "n/a"
            age_days = (datetime.now(timezone.utc) - idea["last_confirmed"]).days if idea.get("last_confirmed") else 0
            lines.append(
                f"- **{idea['name']}** ({idea['status']}) → stale?\n"
                f"  Grund: last-confirmed {lc} ({age_days}d), confidence unter {STALE_THRESHOLD}"
            )

    if sycophancy_flags:
        lines.append("\n## Sycophancy-Flags\n")
        lines.append("*Diese Bewertungen wurden nach unten korrigiert weil Bias erkannt wurde:*\n")
        lines.extend(sycophancy_flags)

    if not status_changes and not connections and not supersession_proposals and not decay_proposals:
        lines.append("\n*Keine Änderungen vorgeschlagen. Alle Ideen stabil.*\n")

    lines.append(f"\n---\n*Archivar v1.1 | Model: gemma3:27b | Sycophancy-Check: qwen2.5:14b | {today}*\n")
    lines.append("<!-- awaiting-approval: ja -->\n")

    proposals_file.write_text("\n".join(lines), encoding="utf-8")
    log(
        f"[PROPOSALS] Written to {proposals_file} "
        f"({len(status_changes)} changes, {len(connections)} connections, "
        f"{len(supersession_proposals)} supersessions, {len(decay_proposals)} decay, "
        f"{len(sycophancy_flags)} flags)"
    )
    return proposals_file


# --- Feedback Integration ---

def get_feedback_summary():
    """Summarize recent feedback for the evaluator."""
    feedback = read_feedback()
    if not feedback:
        return "No previous feedback available."

    lines = []
    for fb in feedback[-10:]:
        lines.append(f"- {fb.get('idea', '?')}: {fb.get('verdict', '?')} — {fb.get('reason', 'no reason')}")
    return "\n".join(lines)


# --- Index Update ---

INDEX_FILE = INBOX_DIR / "INDEX-updated.md"

def update_index():
    """Scan vault and regenerate INDEX.md."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sections = {}

    for md in sorted(VAULT_DIR.rglob("*.md")):
        if md.name.startswith(".") or "/.git/" in str(md):
            continue
        rel = md.relative_to(VAULT_DIR)
        parts = rel.parts

        # Determine section
        if len(parts) == 1:
            section = "Root"
        else:
            section = "/".join(parts[:-1])

        if section not in sections:
            sections[section] = []

        # Read first meaningful line for description
        try:
            text = md.read_text(encoding="utf-8")[:500]
            # Extract status from frontmatter
            status = ""
            status_match = re.search(r"status:\s*(\w+)", text)
            if status_match:
                status = f"**{status_match.group(1)}**"

            # Extract title from first heading
            title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
            title = title_match.group(1).strip() if title_match else md.stem

            entry = f"- `{md.name}`"
            if status:
                entry += f" - {status}"
            if title != md.stem and len(title) < 80:
                entry += f" — {title}"

            sections[section].append(entry)
        except (UnicodeDecodeError, PermissionError):
            sections[section].append(f"- `{md.name}`")

    # Build INDEX.md
    lines = [
        f"# INDEX.md",
        f"*Letztes Update: {today} — automatisch gepflegt durch Archivar*\n",
    ]

    # Root first, then sorted sections
    if "Root" in sections:
        lines.append("## Root")
        lines.extend(sorted(sections.pop("Root")))
        lines.append("")

    for section in sorted(sections.keys()):
        lines.append(f"## {section}/")
        lines.extend(sorted(sections[section]))
        lines.append("")

    INDEX_FILE.write_text("\n".join(lines), encoding="utf-8")
    log(f"[INDEX] Updated INDEX.md ({sum(len(v) for v in sections.values()) + len(sections.get('Root', []))} entries)")


# --- Main ---

def run():
    log("[ARCHIVAR] Starting evaluation run")

    ideas = read_ideas()
    if not ideas:
        log("[ARCHIVAR] No ideas found in vault")
        return

    briefings = read_recent_briefings(days=7)
    briefings_summary = "\n".join(
        f"- {b['name']}: {b['content'][:200]}" for b in briefings[:7]
    )
    session_summary = read_session()
    feedback_summary = get_feedback_summary()
    feedback_history = read_feedback()

    log(f"[ARCHIVAR] Found {len(ideas)} ideas, {len(briefings)} recent briefings")

    evaluations = {}
    for idea in ideas:
        log(f"[EVAL] {idea['name']} (status: {idea['status']})")
        evaluation = evaluate_idea(idea, briefings_summary, session_summary, feedback_summary)
        if evaluation:
            evaluation, was_biased = sycophancy_check(idea, evaluation)
            if was_biased:
                log(f"[SYCOPHANCY] {idea['name']}: bias detected and adjusted")
        evaluations[idea["name"]] = evaluation

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    frontmatter_updates = 0
    for idea in ideas:
        eval_data = evaluations.get(idea["name"])
        confidence = compute_confidence(idea, eval_data, feedback_history)
        if eval_data is not None:
            eval_data["confidence"] = confidence

        externally_confirmed = bool(eval_data and eval_data.get("external", 0) >= 2)
        if externally_confirmed:
            lc_str = today_str
        elif idea.get("last_confirmed"):
            lc_str = idea["last_confirmed"].strftime("%Y-%m-%d")
        else:
            lc_str = today_str

        try:
            if update_idea_frontmatter(idea["path"], confidence, lc_str):
                frontmatter_updates += 1
        except Exception as e:
            log(f"[FRONTMATTER] Failed to update {idea['name']}: {e}")
    log(f"[CONFIDENCE] Updated frontmatter on {frontmatter_updates}/{len(ideas)} ideas")

    supersession_proposals = detect_supersessions(ideas)
    if supersession_proposals:
        log(f"[SUPERSESSION] Found {len(supersession_proposals)} candidate pairs")

    decay_proposals = [idea for idea in ideas if apply_decay_check(idea)]
    if decay_proposals:
        log(f"[DECAY] {len(decay_proposals)} ideas below stale threshold")

    proposals_file = generate_proposals(evaluations, supersession_proposals, decay_proposals)
    update_index()
    log(f"[ARCHIVAR] Run complete. Proposals: {proposals_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        print("Dry run — would evaluate:")
        for idea in read_ideas():
            print(f"  {idea['name']} ({idea['status']}, modified {idea['modified'].strftime('%Y-%m-%d')})")
        print(f"\nBriefings available: {len(read_recent_briefings())}")
        print(f"Feedback entries: {len(read_feedback())}")
    elif len(sys.argv) > 1 and sys.argv[1] == "--index-only":
        update_index()
        print("INDEX.md updated")
    else:
        run()
