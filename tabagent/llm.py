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
import subprocess
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


def ask(prompt: str) -> str:
    tag = OPENAI_MODEL if KEY else MODEL
    h = hashlib.sha256(f"{tag}\x00{prompt}".encode()).hexdigest()[:32]
    f = CACHE / f"{h}.json"
    if f.exists():
        _CALLS["cached"] += 1
        return json.loads(f.read_text())["response"]
    try:
        if KEY:
            txt = _via_openai(prompt)
        elif URL:
            txt = _via_http(prompt)
        else:
            txt = _via_cli(prompt)
    except Exception:
        txt = ""
    if txt:
        _CALLS["n"] += 1
        f.write_text(json.dumps({"model": tag, "prompt": prompt, "response": txt}))
    return txt


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
