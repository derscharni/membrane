#!/usr/bin/env python3
"""
Model Router — Routes requests to the best available model based on
task type, latency requirements, and API health.

Core use case: When Anthropic is down, automatically fall back to
local models via Ollama. Your agent never goes fully offline.

Location: ~/.membrane/model_router.py
Config: ~/.membrane/model-router-config.yaml
"""

import json
import os
import time
import threading
import logging
from dataclasses import dataclass
from pathlib import Path

import requests
import yaml

CONFIG_PATH = Path(os.environ.get("MEMBRANE_DIR", os.path.expanduser("~/.membrane"))) / "model-router-config.yaml"
LOG_FILE = Path(os.environ.get("MEMBRANE_DIR", os.path.expanduser("~/.membrane"))) / "model-router.log"

OLLAMA_URL = "http://localhost:11434"
ANTHROPIC_URL = "https://api.anthropic.com"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ"
)
log = logging.getLogger("model-router")


@dataclass
class ModelStatus:
    name: str
    provider: str  # "ollama" or "anthropic"
    healthy: bool = True
    last_check: float = 0
    last_error: str = ""
    avg_latency: float = 0
    error_count: int = 0


class HealthMonitor:
    """Tracks health of all configured models."""

    def __init__(self, check_interval=60):
        self.models: dict[str, ModelStatus] = {}
        self.check_interval = check_interval
        self._lock = threading.Lock()

    def register(self, name: str, provider: str):
        self.models[name] = ModelStatus(name=name, provider=provider)

    def check_ollama(self, model_name: str) -> bool:
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model_name, "prompt": "hi", "stream": False,
                      "options": {"num_predict": 1}},
                timeout=15
            )
            return resp.status_code == 200
        except Exception:
            return False

    def check_anthropic(self) -> bool:
        try:
            resp = requests.get(
                f"{ANTHROPIC_URL}/v1/models",
                headers={"x-api-key": "dummy", "anthropic-version": "2023-06-01"},
                timeout=10
            )
            # 401 = auth failed but API is up. 529 = overloaded.
            return resp.status_code in (200, 401, 403)
        except Exception:
            return False

    def update(self):
        with self._lock:
            # Check Anthropic once for all Anthropic models
            anthropic_healthy = self.check_anthropic()

            for name, status in self.models.items():
                if status.provider == "anthropic":
                    was_healthy = status.healthy
                    status.healthy = anthropic_healthy
                    if not anthropic_healthy:
                        status.error_count += 1
                        status.last_error = "API unreachable or overloaded"
                    if was_healthy and not status.healthy:
                        log.warning(f"[health] {name} DOWN — falling back to local models")
                    elif not was_healthy and status.healthy:
                        log.info(f"[health] {name} RECOVERED")

                elif status.provider == "ollama":
                    ollama_model = name.split("/", 1)[1] if "/" in name else name
                    was_healthy = status.healthy
                    status.healthy = self.check_ollama(ollama_model)
                    if not status.healthy:
                        status.error_count += 1
                        status.last_error = "Ollama model not responding"
                    if was_healthy != status.healthy:
                        state = "UP" if status.healthy else "DOWN"
                        log.info(f"[health] {name} {state}")

                status.last_check = time.time()

    def is_healthy(self, model_name: str) -> bool:
        with self._lock:
            if model_name in self.models:
                return self.models[model_name].healthy
            return False

    def get_status(self) -> dict:
        with self._lock:
            return {
                name: {
                    "healthy": s.healthy,
                    "provider": s.provider,
                    "errors": s.error_count,
                    "last_error": s.last_error,
                }
                for name, s in self.models.items()
            }


class ModelRouter:
    """Routes tasks to the best available model."""

    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config = self._load_config(config_path)
        self.health = HealthMonitor(check_interval=self.config.get("health_check_interval", 60))
        self._register_models()
        self._monitor_thread = None

    def _load_config(self, path: Path) -> dict:
        if path.exists():
            return yaml.safe_load(path.read_text()) or {}
        return self._default_config()

    def _default_config(self) -> dict:
        return {
            "health_check_interval": 60,
            "rules": [
                {"task": "conversation", "latency": "medium",
                 "model": "anthropic/claude-sonnet-4-6",
                 "fallback": "ollama/gemma3:27b"},
                {"task": "classification", "latency": "low",
                 "model": "ollama/qwen2.5:14b",
                 "fallback": "ollama/gemma4:e4b"},
                {"task": "fast-filter", "latency": "realtime",
                 "model": "ollama/gemma4:e4b",
                 "fallback": None},
                {"task": "deep-research", "latency": "medium",
                 "model": "anthropic/claude-sonnet-4-6",
                 "fallback": "ollama/gemma3:27b"},
                {"task": "archival", "latency": "any",
                 "model": "ollama/gemma3:27b",
                 "fallback": None},
                {"task": "cron-briefing", "latency": "medium",
                 "model": "anthropic/claude-sonnet-4-6",
                 "fallback": "ollama/gemma3:27b"},
            ],
            "fallback": "ollama/gemma3:27b",
        }

    def _register_models(self):
        seen = set()
        for rule in self.config.get("rules", []):
            # Primary + legacy single fallback
            for key in ("model", "fallback"):
                model = rule.get(key)
                if model and model not in seen:
                    provider = model.split("/")[0]
                    self.health.register(model, provider)
                    seen.add(model)
            # Multi-fallback chain
            for model in rule.get("fallbacks", []) or []:
                if model and model not in seen:
                    provider = model.split("/")[0]
                    self.health.register(model, provider)
                    seen.add(model)
        # Global fallback
        fb = self.config.get("fallback")
        if fb and fb not in seen:
            provider = fb.split("/")[0]
            self.health.register(fb, provider)

    def start_monitoring(self):
        """Start background health monitoring."""
        self.health.update()  # Initial check

        def _loop():
            while True:
                time.sleep(self.health.check_interval)
                self.health.update()

        self._monitor_thread = threading.Thread(target=_loop, daemon=True)
        self._monitor_thread.start()
        log.info("[router] Health monitoring started")

    def route(self, task: str) -> str | None:
        """Return the best available model for a given task.

        Priority order:
        1. rule.model (primary)
        2. rule.fallbacks (list, in order) — for multi-fallback chains
        3. rule.fallback (single, legacy) — kept for backward compat
        4. config.fallback (global)
        """
        # Find matching rule
        rule = None
        for r in self.config.get("rules", []):
            if r["task"] == task:
                rule = r
                break

        if not rule:
            log.info(f"[route] No rule for task '{task}', using global fallback")
            return self._resolve_fallback(self.config.get("fallback"))

        primary = rule["model"]

        if self.health.is_healthy(primary):
            log.info(f"[route] {task} → {primary}")
            return primary

        # Multi-fallback chain (new): rule.fallbacks = [model1, model2, ...]
        for fb in rule.get("fallbacks", []) or []:
            if fb and self.health.is_healthy(fb):
                log.warning(f"[route] {task} → {fb} (from fallbacks chain, primary {primary} unhealthy)")
                return fb

        # Legacy single fallback
        single_fb = rule.get("fallback")
        if single_fb and self.health.is_healthy(single_fb):
            log.warning(f"[route] {task} → {single_fb} (primary {primary} unhealthy)")
            return single_fb

        # Global fallback
        global_fb = self.config.get("fallback")
        if global_fb and self.health.is_healthy(global_fb):
            log.warning(f"[route] {task} → {global_fb} (global fallback)")
            return global_fb

        log.error(f"[route] {task} → NO MODEL AVAILABLE")
        return None

    def _resolve_fallback(self, model: str | None) -> str | None:
        if model and self.health.is_healthy(model):
            return model
        return None

    def get_ollama_model_name(self, routed_model: str) -> str:
        """Extract Ollama model name from router format."""
        if "/" in routed_model:
            return routed_model.split("/", 1)[1]
        return routed_model

    def is_local(self, model: str) -> bool:
        return model.startswith("ollama/")

    def is_api(self, model: str) -> bool:
        return model.startswith("anthropic/")

    def status(self) -> dict:
        return {
            "models": self.health.get_status(),
            "rules": len(self.config.get("rules", [])),
        }


def generate_local(model: str, prompt: str, options: dict | None = None) -> str | None:
    """Send a prompt to a local Ollama model and return the response."""
    ollama_model = model.split("/", 1)[1] if "/" in model else model
    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
    }
    if options:
        payload["options"] = options
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json().get("response", "")
    except Exception as e:
        log.error(f"[generate] {model} error: {e}")
    return None


# --- CLI ---

def main():
    """CLI: show status or route a task."""
    import sys

    router = ModelRouter()
    router.health.update()

    if len(sys.argv) < 2 or sys.argv[1] == "status":
        print("Model Router Status")
        print("=" * 50)
        for name, info in router.status()["models"].items():
            icon = "OK" if info["healthy"] else "DOWN"
            print(f"  [{icon:4s}] {name} ({info['provider']}) errors={info['errors']}")
        print(f"\nRules: {router.status()['rules']}")

    elif sys.argv[1] == "route":
        if len(sys.argv) < 3:
            print("Usage: model_router.py route <task>")
            print("Tasks: conversation, classification, fast-filter, deep-research, archival, cron-briefing")
            sys.exit(1)
        task = sys.argv[2]
        model = router.route(task)
        if model:
            print(f"{task} → {model}")
        else:
            print(f"{task} → NO MODEL AVAILABLE")
            sys.exit(1)

    elif sys.argv[1] == "health":
        router.health.update()
        for name, info in router.status()["models"].items():
            icon = "OK" if info["healthy"] else "DOWN"
            err = f" ({info['last_error']})" if info["last_error"] and not info["healthy"] else ""
            print(f"  [{icon:4s}] {name}{err}")

    else:
        print("Usage: model_router.py [status|route <task>|health]")


if __name__ == "__main__":
    main()
