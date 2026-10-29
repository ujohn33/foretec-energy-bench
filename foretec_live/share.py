"""Share links for the leaderboard: GET /s/<token> (a page with the link-preview tags LinkedIn reads, which sends people
on to the leaderboard view) and GET /s/<token>.png (the 1200 x 627 card of that view's ranking, rendered once).

The token is the leaderboard's link parameters, base64url-encoded, plus the data version it was shared at:
  <base64url of "z=BE&t=price&r=d7&c=all&e=0&a=mean&m=t0_gamma_cov">.<YYYYMMDDHHMM>
Only known values are accepted (zones, targets and models from the site's own summary.json), and the image is made
by a headless Chromium opening the site's own card view (index.html?card=1&...), so nothing but a ranking the site
itself computes can be rendered or served. Images are cached in $STATE_DIRECTORY; renders are serialised and limited
per address. It listens on 127.0.0.1 behind Caddy (scripts/systemd/foretec-share.service).
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode

HOST, PORT = "127.0.0.1", int(os.environ.get("SHARE_PORT", "8092"))
SITE = os.environ.get("SHARE_SITE", "https://energybench.foretec.co")
SUMMARY = Path(os.environ.get("SHARE_SUMMARY", "/srv/foretec/foretec-live/site/data/summary.json"))
CACHE = Path(os.environ.get("STATE_DIRECTORY", "/var/lib/foretec-share"))
DISPLAY = {"tso_cutoff": "TSO_cutoff", "tso_final": "TSO_final"}   # as the site shows them (app.js DISPLAY)
RATED = {"d7": "the last 7 days", "d30": "the last 30 days", "all": "all live days", "bt": "the backtest"}
COV, AGG = {"all", "with", "without"}, {"mean", "median", "gmean"}
DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TOKEN = re.compile(r"^([A-Za-z0-9_-]{8,400})\.(\d{8,14})(\.png)?$")
RENDERS_PER_IP_HOUR, MAX_CACHE_FILES, KEEP_DAYS = 30, 5000, 90

_render_lock = threading.Lock()
_hits: dict[str, list[float]] = {}
_known = {"t": 0.0, "zones": set(), "targets": {}, "models": set(), "labels": {}}


def known() -> dict:
    """Zones, targets and models the site knows, re-read from summary.json at most every 10 minutes."""
    if time.time() - _known["t"] > 600:
        d = json.loads(SUMMARY.read_text())
        _known.update(t=time.time(), zones=set(d["config"]["zones"]), targets=d["config"]["targets"],
                      models={m["name"] for m in d["models"]}, labels=d["config"].get("zone_labels") or {})
    return _known


def parse(token: str) -> tuple[dict, str, bool] | None:
    """Validated parameters, data version and whether the image was asked for; None for anything unknown."""
    m = TOKEN.match(token)
    if not m:
        return None
    try:
        raw = base64.urlsafe_b64decode(m.group(1) + "=" * (-len(m.group(1)) % 4)).decode()
    except Exception:  # noqa: BLE001 - not a token
        return None
    q = {k: v[0] for k, v in parse_qs(raw, max_num_fields=8).items()}
    k = known()
    if q.get("p") == "x":   # a forecast-explorer day: zone, target, issue day (its day page must exist), optional model
        p = {"p": "x", "d": q.get("d", ""), "z": q.get("z", ""), "t": q.get("t", "")}
        if not DAY.match(p["d"]) or not (SUMMARY.parent / "days" / f"{p['d']}.json").exists() \
                or p["z"] not in k["zones"] or p["t"] not in k["targets"]:
            return None
        if q.get("m"):
            if q["m"] not in k["models"]:
                return None
            p["m"] = q["m"]
        return p, m.group(2), bool(m.group(3))
    p = {"z": q.get("z", "all"), "t": q.get("t", "all"), "r": q.get("r", "d7"), "c": q.get("c", "all"),
         "e": q.get("e", "0"), "a": q.get("a", "mean")}
    if (p["z"] != "all" and p["z"] not in k["zones"]) or (p["t"] != "all" and p["t"] not in k["targets"]) \
            or p["r"] not in RATED or p["c"] not in COV or p["e"] not in ("0", "1") or p["a"] not in AGG:
        return None
    if q.get("m"):
        if q["m"] not in k["models"]:
            return None
        p["m"] = q["m"]
    return p, m.group(2), bool(m.group(3))


def page_url(p: dict, card: bool = False) -> str:
    """The view a token stands for (and its card for the renderer)."""
    if p.get("p") == "x":
        q = {"day": p["d"], "zone": p["z"], "target": p["t"], **({"m": p["m"]} if p.get("m") else {})}
        return f"{SITE}/explorer.html?{urlencode(q)}" + ("&card=1" if card else "")
    return f"{SITE}/?{urlencode(p)}" + ("&card=1" if card else "")


def describe(p: dict) -> tuple[str, str]:
    k = known()
    if p.get("p") == "x":
        from datetime import date, timedelta
        d = date.fromisoformat(p["d"]) + timedelta(days=1)
        tl = {"price": "day-ahead price", "wind": "wind", "solar": "solar", "load": "load"}
        what = f"{k['labels'].get(p['z'], p['z'])} {tl.get(p['t'], p['t'])}, {d.day} {d.strftime('%b %Y')}"
        title = f"{DISPLAY.get(p['m'], p['m'])} on {what}" if p.get("m") else f"{what}: day-ahead forecasts vs actual"
        return title, ("Every model's day-ahead forecast, locked before the 12:00 gate, against what actually happened. "
                       "EnergyBench by Foretec, the live open benchmark for energy forecasts.")
    zone = "all zones" if p["z"] == "all" else k["labels"].get(p["z"], p["z"])
    tl = {"price": "Day-ahead price", "wind": "Wind", "solar": "Solar", "load": "Load"}
    target = "All targets" if p["t"] == "all" else tl.get(p["t"], p["t"])
    view = f"{target} · {zone} · {RATED[p['r']]}"
    title = f"How {DISPLAY.get(p['m'], p['m'])} ranks: {view}" if p.get("m") else f"{view} · EnergyBench by Foretec"
    desc = ("A live, open benchmark of day-ahead energy forecasts: every model forecasts from the data public at 11:30, "
            "locked before the 12:00 gate, and is scored against the grid operators.")
    return title, desc


def _limited(ip: str) -> bool:
    now = time.time()
    hits = [t for t in _hits.get(ip, []) if now - t < 3600]
    _hits[ip] = hits + [now]
    return len(hits) >= RENDERS_PER_IP_HOUR


def _prune() -> None:
    files = sorted(CACHE.glob("*.png"), key=lambda f: f.stat().st_mtime)
    for f in files:
        if len(files) > MAX_CACHE_FILES or time.time() - f.stat().st_mtime > KEEP_DAYS * 86400:
            f.unlink(missing_ok=True)
            files = files[1:]


def render(p: dict, out: Path) -> None:
    from playwright.sync_api import sync_playwright
    url = page_url(p, card=True)
    tmp = out.with_suffix(".tmp.png")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--disable-dev-shm-usage"])
        try:
            page = browser.new_page(viewport={"width": 1200, "height": 627}, device_scale_factor=2)
            page.goto(url, wait_until="networkidle", timeout=45000)
            page.wait_for_selector("body[data-card-ready]", state="attached", timeout=30000)   # no eval: the site's CSP forbids it
            page.locator("#share-card").screenshot(path=str(tmp))
        finally:
            browser.close()
    tmp.replace(out)


class Handler(BaseHTTPRequestHandler):
    server_version = "foretec-share"

    def log_message(self, fmt, *args):   # no addresses in the journal
        pass

    def _send(self, status: int, body: bytes, ctype: str, cache: str = "no-store") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if not path.startswith("/s/"):
            return self._send(404, b"not found", "text/plain")
        got = parse(path[3:])
        if not got:
            return self._send(404, b"unknown share link", "text/plain")
        p, version, want_png = got
        key = hashlib.sha1((urlencode(sorted(p.items())) + version).encode()).hexdigest()[:24]
        if want_png:
            out = CACHE / f"{key}.png"
            if not out.exists():
                ip = self.headers.get("X-Forwarded-For", self.client_address[0]).split(",")[0].strip()
                if _limited(ip):
                    return self._send(429, b"too many images, try again later", "text/plain")
                with _render_lock:
                    if not out.exists():
                        try:
                            render(p, out)
                            _prune()
                        except Exception:  # noqa: BLE001 - the page still works without its image
                            return self._send(503, b"the image could not be rendered", "text/plain")
            return self._send(200, out.read_bytes(), "image/png", "public, max-age=604800, immutable")
        title, desc = describe(p)
        token = path[3:]
        view = page_url(p)
        img = f"{SITE}/s/{token}.png"
        e = lambda s: html.escape(s, quote=True)  # noqa: E731
        body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{e(title)}</title><meta name="description" content="{e(desc)}">
<meta property="og:type" content="website"><meta property="og:site_name" content="EnergyBench by Foretec">
<meta property="og:title" content="{e(title)}"><meta property="og:description" content="{e(desc)}">
<meta property="og:url" content="{e(SITE + '/s/' + token)}"><meta property="og:image" content="{e(img)}">
<meta property="og:image:width" content="2400"><meta property="og:image:height" content="1254">
<meta name="twitter:card" content="summary_large_image"><meta name="twitter:image" content="{e(img)}">
<meta name="robots" content="noindex, follow"><link rel="canonical" href="{e(view)}"><meta http-equiv="refresh" content="0; url={e(view)}">
</head><body><p><a href="{e(view)}">Open the leaderboard</a></p></body></html>"""
        return self._send(200, body.encode(), "text/html; charset=utf-8", "public, max-age=300")


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
