# membrane

A sovereign context layer for AI agents.

Every token that reaches your AI should be:
1. **Safe** — sanitized of hidden Unicode payloads
2. **Compressed** — stripped of redundant history and boilerplate
3. **Routed** — sent to the right model for the task
4. **Archived** — tracked with confidence scores that decay over time

membrane is a collection of small, dependency-light Python modules that sit
between your agent and your LLM provider. Each module does one thing.
All together, they give you context sovereignty.

## Modules

| Module | What it does |
|---|---|
| sanitizer.py | Strips hidden Unicode from agent inputs |
| context_compressor.py | Reduces token usage in long sessions (40-60%) |
| local_worker.py | Runs slow tasks on local models via Ollama |
| model_router.py | Routes tasks to the right model with health checks and fallback chains |
| archivar.py | Tracks ideas with confidence scores, supersession detection, and decay |

## Quick Start

Each module works standalone. Drop it into your project and import:

```python
# Sanitize agent input
from sanitizer import sanitize
cleaned, found = sanitize(text)

# Compress a long message history
from context_compressor import compress_messages
compressed, stats = compress_messages(messages)

# Route a task to the best available model
from model_router import ModelRouter
router = ModelRouter()
model = router.route("classification")

# Queue a task for local Ollama execution
# (write a YAML-frontmatter .md file to the queue directory)
```

## Philosophy

- No frameworks. No microservices. Just Python files you can read.
- Each module works standalone or together.
- Fork it. Modify it. Make it yours.
- Zero external dependencies for core modules (sanitizer, compressor). Minimal dependencies for others (requests for model_router and local_worker).

## Configuration

Modules use environment variables and config files, not hardcoded paths:

```bash
export MEMBRANE_DIR=~/.membrane
export VAULT_DIR=~/.napkin
export OLLAMA_URL=http://localhost:11434
```

See each module's docstring for its specific configuration.

## Part of the Trust Stack

membrane implements the infrastructure layer of the [Trust Stack](https://github.com/derscharni/trust-stack):

- **Layer 1 (Security):** sanitizer.py
- **Layer 2 (Authorization):** [context-handshake](https://github.com/derscharni/context-handshake)
- **Layer 3 (Temporal Integrity):** archivar.py (confidence scoring + decay)

## Fork the idea

See [IDEA.md](IDEA.md)
