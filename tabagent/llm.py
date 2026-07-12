"""LLM backend, in order of preference:

  1. OpenAI, called straight from the GPU box, if a key is on disk. This is the one that
     works unattended: no SSH reverse tunnel to drop, and calls can go in parallel.
  2. The tunnel back to the laptop's `claude` CLI (llm_server.py). Fine, but every time the
     SSH session died it silently returned empty strings and burned a whole round.
  3. The local `claude` CLI, when running on the laptop.

Every response is cached on disk by hash(model, prompt), so a rerun costs nothing and a
crashed run resumes exactly where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

CACHE = Path(__file__).parent / "cache" / "llm"
CACHE.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("TABAGENT_LLM", "claude-opus-4-8")
URL = os.environ.get("TABAGENT_LLM_URL")  # set on the server -> use the tunnel
_CALLS = {"n": 0, "cached": 0}

OPENAI_MODEL = os.environ.get("TABAGENT_OPENAI_MODEL", "gpt-4.1")
_ENV_PATHS = [Path.home() / "ralphton" / ".env", Path(__file__).parent / ".env"]


def _openai_key() -> str | None:
    k = os.environ.get("OPENAI_API_KEY")
    if k:
        return k.strip()
    for p in _ENV_PATHS:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if "openai_api_key" in line.lower() and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


KEY = _openai_key()


def _via_openai(prompt: str) -> str:
    body = json.dumps({
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"content-type": "application/json",
                 "authorization": f"Bearer {KEY}"},
    )
    with urllib.request.urlopen(req, timeout=240) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"].strip()


def n_calls() -> int:
    return _CALLS["n"]


def reset_calls() -> None:
    _CALLS.update(n=0, cached=0)


def _via_http(prompt: str) -> str:
    req = urllib.request.Request(
        URL, data=json.dumps({"prompt": prompt}).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=330) as r:
        return json.loads(r.read())["response"]


def _via_cli(prompt: str) -> str:
    out = subprocess.run(
        ["claude", "-p", "--model", MODEL, prompt],
        capture_output=True, text=True, timeout=300, cwd="/tmp",
    )
    return out.stdout.strip()


def backend() -> str:
    return "openai" if KEY else ("tunnel" if URL else "claude-cli")


class LLMDown(RuntimeError):
    """No backend answered. Raised, never swallowed.

    This class exists because the alternative cost us a full-scale run. `ask` used to catch
    every exception and return "", and the caller read "" as "the agent had no proposal" and
    quietly substituted a random prior. The OpenAI key was revoked mid-experiment, so all
    three arms silently became random search while still reporting themselves as `agent` and
    `anchored`. Nothing crashed; the numbers looked plausible; they were fiction.

    A dead LLM must stop the experiment, not redefine it.
    """


def _backends():
    """Every backend that could possibly answer, best first, each with the model tag its
    answers are cached under. We try them all before giving up — a revoked key should fall
    through to the tunnel, not kill the run.

    The tag is part of the cache key because two backends are two different models, and a
    cached gpt-4.1 answer must not be replayed as if Claude had said it."""
    if KEY:
        yield "openai", OPENAI_MODEL, _via_openai
    if URL:
        yield "tunnel", MODEL, _via_http
    if shutil.which("claude"):
        yield "claude-cli", MODEL, _via_cli


def _cached(tag: str, prompt: str) -> Path:
    h = hashlib.sha256(f"{tag}\x00{prompt}".encode()).hexdigest()[:32]
    return CACHE / f"{h}.json"


def ask(prompt: str, retries: int = 2) -> str:
    global KEY
    # A hit under ANY backend's tag is a hit — the run should resume for free even if it
    # comes back up on a different backend than it went down on.
    for _, tag, _ in _backends():
        f = _cached(tag, prompt)
        if f.exists():
            _CALLS["cached"] += 1
            return json.loads(f.read_text())["response"]

    errs = []
    for name, tag, fn in _backends():
        for attempt in range(retries + 1):
            try:
                txt = fn(prompt).strip()
                if txt:
                    _CALLS["n"] += 1
                    _cached(tag, prompt).write_text(
                        json.dumps({"model": tag, "prompt": prompt, "response": txt}))
                    return txt
                errs.append(f"{name}: empty response")
            except Exception as e:
                errs.append(f"{name}: {type(e).__name__}: {e}")
                # A revoked key will never come back within a run. Demote it once and let
                # the tunnel take over, instead of paying two timeouts on every call.
                if name == "openai" and "401" in str(e):
                    print("  [llm] OpenAI key rejected (401) — dropping it, "
                          "falling through to the next backend", flush=True)
                    KEY = None
                    break
                if attempt < retries:
                    time.sleep(2 ** attempt)
    raise LLMDown("no LLM backend answered:\n  " + "\n  ".join(errs[-6:]))


def preflight() -> str:
    """Prove the LLM is alive BEFORE spending GPU hours. Returns the backend that answered.

    Call this at the top of anything that depends on the agent. A run that cannot reach its
    agent must never start."""
    names = [n for n, _, _ in _backends()]
    if not names:
        raise LLMDown("no backend configured: no OPENAI_API_KEY, no TABAGENT_LLM_URL, "
                      "no `claude` on PATH")
    txt = ask(f"Reply with exactly the word PONG. (nonce {os.getpid()})", retries=1)
    if "PONG" not in txt.upper():
        raise LLMDown(f"backend answered but not sanely: {txt[:120]!r}")
    return ",".join(names)


def ask_json(prompt: str, default):
    raw = ask(prompt + "\n\nRespond with ONLY valid JSON. No prose, no code fences.")
    if not raw:
        return default
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    if m:
        raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        for o, c in (("{", "}"), ("[", "]")):
            i, j = raw.find(o), raw.rfind(c)
            if 0 <= i < j:
                try:
                    return json.loads(raw[i : j + 1])
                except json.JSONDecodeError:
                    pass
    return default
