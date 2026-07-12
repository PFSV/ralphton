"""LLM backend. Talks to the local `claude` CLI directly, or over HTTP when the
experiment is running on the GPU box (see llm_server.py + the SSH reverse tunnel).
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


def ask(prompt: str) -> str:
    key = hashlib.sha256(f"{MODEL}\x00{prompt}".encode()).hexdigest()[:32]
    f = CACHE / f"{key}.json"
    if f.exists():
        _CALLS["cached"] += 1
        return json.loads(f.read_text())["response"]
    try:
        txt = _via_http(prompt) if URL else _via_cli(prompt)
    except Exception:
        txt = ""
    if txt:
        _CALLS["n"] += 1
        f.write_text(json.dumps({"prompt": prompt, "response": txt}))
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
