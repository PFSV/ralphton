"""Local LLM service. The A100 reaches it through an SSH reverse tunnel.

The experiment runs on the GPU box (TabICL forward passes), but the agent's brain is the
`claude` CLI, which only exists on the laptop. So the laptop serves it over HTTP and the
server calls back through -R. Every response is cached on disk by prompt hash, which makes
the whole multi-hour run resumable and replayable at zero LLM cost.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CACHE = Path(__file__).parent / "cache" / "llm"
CACHE.mkdir(parents=True, exist_ok=True)
MODEL = "claude-opus-4-8"
STATS = {"hit": 0, "miss": 0, "fail": 0}


def complete(prompt: str) -> str:
    key = hashlib.sha256(f"{MODEL}\x00{prompt}".encode()).hexdigest()[:32]
    f = CACHE / f"{key}.json"
    if f.exists():
        STATS["hit"] += 1
        return json.loads(f.read_text())["response"]
    try:
        out = subprocess.run(
            ["claude", "-p", "--model", MODEL, prompt],
            capture_output=True, text=True, timeout=300, cwd="/tmp",
        )
        txt = out.stdout.strip()
    except subprocess.TimeoutExpired:
        txt = ""
    if txt:
        STATS["miss"] += 1
        f.write_text(json.dumps({"prompt": prompt, "response": txt}))
    else:
        STATS["fail"] += 1
    print(f"  llm hit={STATS['hit']} miss={STATS['miss']} fail={STATS['fail']}", flush=True)
    return txt


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        resp = complete(body.get("prompt", ""))
        payload = json.dumps({"response": resp, "billed": STATS["miss"]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"llm_server on :{port} (model={MODEL}, cache={CACHE})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
