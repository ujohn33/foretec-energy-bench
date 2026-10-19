"""Waiting list for the bidding competition (compete.html): POST /api/waitlist, appended to a JSON-lines file.

Standard library only, so it runs on the system Python with nothing imported from the benchmark. It listens on
127.0.0.1 behind Caddy, which terminates HTTPS and caps the body; systemd runs it as a throwaway user with only its
state directory writable (scripts/systemd/foretec-waitlist.service).

  accepted   email (checked, lower-cased, at most 254 characters), role (one of ROLES), consent (must be true)
  stored     {"t": UTC time, "email", "role"}, one line each, in $STATE_DIRECTORY/waitlist.jsonl; an email already on
             the list is acknowledged and not stored twice
  refused    a filled-in honeypot field (silently accepted, not stored), more than PER_IP_HOUR requests an hour from one
             address, more than PER_DAY new entries a day in total
Reading the list (as root):  cat /var/lib/foretec-waitlist/waitlist.jsonl
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote

HOST, PORT = "127.0.0.1", int(os.environ.get("WAITLIST_PORT", "8091"))
STORE = Path(os.environ.get("STATE_DIRECTORY", "/var/lib/foretec-waitlist")) / "waitlist.jsonl"
ROLES = {"participant", "sponsor", "researcher", "curious"}
EMAIL = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$")
MAX_BODY, PER_IP_HOUR, PER_DAY = 2048, 10, 500
PAGE = "/compete.html"
# foretec.be posts its sign-up form here too (cross-site): only these origins may read the reply
ORIGINS = {"https://foretec.be", "https://www.foretec.be"}

_lock = threading.Lock()
_hits: dict[str, list[float]] = {}
_day = {"date": "", "n": 0}


def _known(email: str) -> bool:
    if not STORE.exists():
        return False
    with STORE.open() as f:
        return any(json.loads(line).get("email") == email for line in f if line.strip())


def _limited(ip: str) -> bool:
    now = time.time()
    with _lock:
        hits = [t for t in _hits.get(ip, []) if now - t < 3600]
        hits.append(now)
        _hits[ip] = hits
        if len(_hits) > 10000:   # forget idle addresses
            for k in [k for k, v in _hits.items() if now - v[-1] > 3600]:
                del _hits[k]
        return len(hits) > PER_IP_HOUR


def add(data: dict) -> tuple[int, str]:
    """(HTTP status, message). Never says whether an address was already on the list."""
    if str(data.get("website", "")).strip():                  # honeypot: bots fill every field
        return 200, "ok"
    email = str(data.get("email", "")).strip().lower()
    role = str(data.get("role", "curious")).strip()
    consent = data.get("consent") in (True, "yes", "true", "on")
    if len(email) > 254 or not EMAIL.match(email):
        return 400, "Please enter a valid email address."
    if role not in ROLES:
        return 400, "Please choose one of the options."
    if not consent:
        return 400, "Please tick the box so we may email you."
    with _lock:
        today = datetime.now(timezone.utc).date().isoformat()
        if _day["date"] != today:
            _day.update(date=today, n=0)
        if _known(email):
            return 200, "ok"
        if _day["n"] >= PER_DAY:
            return 429, "Too many sign-ups today, please try again tomorrow."
        STORE.parent.mkdir(parents=True, exist_ok=True)
        with STORE.open("a") as f:
            f.write(json.dumps({"t": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "email": email, "role": role}) + "\n")
        os.chmod(STORE, 0o600)
        _day["n"] += 1
    return 200, "ok"


class Handler(BaseHTTPRequestHandler):
    server_version = "waitlist"
    sys_version = ""

    def log_message(self, fmt, *args):   # no addresses or emails in the journal
        pass

    def _cors(self):
        origin = self.headers.get("Origin", "")
        if origin in ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def do_OPTIONS(self):   # CORS preflight from foretec.be
        self.send_response(204 if self.headers.get("Origin", "") in ORIGINS else 403)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "POST")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _reply(self, status: int, msg: str, as_json: bool):
        if as_json:
            body = json.dumps({"ok": status == 200} if status == 200 else {"error": msg}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.end_headers()
            self.wfile.write(body)
        else:                             # plain form post (no JavaScript): back to the page with the outcome
            self.send_response(303)
            self.send_header("Location", f"{PAGE}?joined=1#join" if status == 200 else f"{PAGE}?error={quote(msg)}#join")
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_POST(self):
        as_json = "application/json" in (self.headers.get("Accept", "") + self.headers.get("Content-Type", ""))
        if self.path.split("?")[0] != "/api/waitlist":
            return self._reply(404, "Not found.", as_json)
        ip = (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[0].strip()
        if _limited(ip):
            return self._reply(429, "Too many attempts, please try again in an hour.", as_json)
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = -1
        if not 0 < n <= MAX_BODY:
            return self._reply(400, "Invalid request.", as_json)
        raw = self.rfile.read(n).decode("utf-8", "replace")
        try:
            if "application/json" in self.headers.get("Content-Type", ""):
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError
            else:
                data = {k: v[0] for k, v in parse_qs(raw).items()}
        except ValueError:
            return self._reply(400, "Invalid request.", as_json)
        self._reply(*add(data), as_json)

    def do_GET(self):
        self._reply(405, "Use the form on the competition page.", True)


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
