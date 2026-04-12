# IDEA.md

Inspired by Andrej Karpathy's "fork the idea, not the code" pattern.

This file exists so you can copy the idea into your own LLM context, your wiki, or your agent setup - without cloning the code.

---

## The idea in this repo: membrane

Your AI agent's context is the most important thing it has. Most systems treat it as an implementation detail.

membrane treats it as a first-class concern.

Five questions every agent deployment should answer:
1. Is the input sanitized? (sanitizer.py)
2. Is old context compressed away? (context_compressor.py)
3. Is each task going to the right model? (model_router.py)
4. Are slow tasks running locally? (local_worker.py)
5. Does your knowledge decay gracefully? (archivar.py)

---

## Copy this into your LLM context

Every token your agent sees should pass through four gates:
- Gate 1: Sanitize — strip hidden Unicode, zero-width chars, steganographic payloads
- Gate 2: Compress — prune ACK messages, truncate old tool outputs, enforce rolling windows
- Gate 3: Route — send classification tasks to small local models, creative tasks to large cloud models
- Gate 4: Archive — score every piece of knowledge with confidence that decays over time

The question to ask: "If I removed this token from the context, would the agent notice?"

---

## Go deeper

github.com/derscharni/membrane
Part of: github.com/derscharni/trust-stack
Inspired by: gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
