"""Render the site's default link preview, foretec_live/web/og-card.png: the look of the share cards (share.py), with one
real forecast from site/data/days. Run where the site is built, with the share service's Playwright:

  PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright /srv/foretec/share-venv/bin/python scripts/og_card.py /tmp/og-card.png

The card is a 1200 x 630 page at 2x. Bump the ?v= of og:image in seo.py when it changes: previews are cached by URL.
"""
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISSUE, SERIES, MODEL = "2026-10-29", "DE_solar", "t0_gamma_cov"
LABEL = "Solar power in Germany"
W, H, TOP = 480, 236, 14

PAGE = """<!doctype html><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600&family=Inter:ital,wght@1,500..900&display=swap">
<style>
html,body{margin:0}
#card{width:1200px; height:630px; box-sizing:border-box; padding:46px 56px 34px; background:#F3F8FA; color:#0B1F3A; display:flex; flex-direction:column;
  font-family:"Source Sans 3",sans-serif; overflow:hidden}
.head{display:flex; justify-content:space-between; align-items:flex-start}
.lockup{display:flex; flex-direction:column; font-family:Inter,sans-serif; font-style:italic; line-height:1}
.product{font-size:50px; font-weight:600; letter-spacing:-.032em; color:#3F6C8B}
.by{margin-top:7px; padding-left:.12em; font-size:19px; font-weight:500; letter-spacing:-.01em; color:#4E6578}
.by b{font-weight:900; letter-spacing:-.032em; color:#0B1F3A}
.live{display:inline-flex; align-items:center; gap:9px; margin-top:6px; color:#D7263D; font-size:22px; font-weight:600}
.live i{width:12px; height:12px; border-radius:50%; background:#D7263D}
.body{flex:1; display:flex; align-items:center; justify-content:space-between; gap:48px}
h1{margin:0; font-family:Inter,sans-serif; font-style:italic; font-weight:800; font-size:54px; line-height:1.08; letter-spacing:-.03em; text-wrap:balance}
p{margin:18px 0 0; font-size:27px; line-height:1.3; color:#4E6578}
.left{flex:1; min-width:0}
.panel{flex:none; width:520px; box-sizing:border-box; padding:16px 20px 12px; background:#fff; border:1px solid #C6D9E0}
.panel svg{display:block; width:100%}
.cap{display:flex; justify-content:space-between; font-size:19px; color:#4E6578; margin-bottom:6px}
.cap b{color:#0B1F3A; font-weight:600}
.key{display:flex; gap:22px; margin-top:6px; font-size:19px; color:#4E6578}
.key span{display:inline-flex; align-items:center; gap:8px}
.key i{width:22px; height:0; border-top:3px solid}
.key .b{width:18px; height:12px; border:0; background:#DCE9EF}
.foot{display:flex; justify-content:space-between; padding-top:16px; border-top:1px solid #C6D9E0; font-size:23px; font-weight:600; color:#4E6578}
.foot b{font-weight:600; color:#3F6C8B}
</style>
<div id="card">
  <div class="head">
    <div class="lockup"><span class="product">EnergyBench</span><span class="by">by <b>Foretec</b></span></div>
    <span class="live"><i></i>Live</span>
  </div>
  <div class="body">
    <div class="left">
      <h1>The live <span style="white-space:nowrap">day-ahead</span> forecasting benchmark</h1>
      <p>Every model commits before the 12:00 market deadline, then is scored on what actually happened.</p>
    </div>
    <div class="panel">
      <div class="cap"><b>__LABEL__</b><span>__DAY__</span></div>
      __SVG__
      <div class="key"><span><i style="border-color:#0B1F3A"></i>Actual</span><span><i style="border-color:#3F6C8B"></i>Forecast</span><span><i class="b"></i>Forecast range</span></div>
    </div>
  </div>
  <div class="foot"><span>Price · Wind · Solar · Load</span><span>BE · NL · FR · DE-LU · DK1 · DK2</span><b>energybench.foretec.co</b></div>
</div>
"""


def chart(ser: dict, model: str) -> str:
    m = ser["models"][model]
    act, p, lo, hi = ser["actual"], m["p"], m["lo"], m["hi"]
    top = max(*act, *hi)
    pts = lambda seq: " ".join(f"{W * i / (len(act) - 1):.1f},{TOP + (H - TOP) * (1 - v / top):.1f}" for i, v in seq)
    line = lambda vals, col: f'<polyline points="{pts(enumerate(vals))}" fill="none" stroke="{col}" stroke-width="2.5" stroke-linejoin="round"/>'
    return (f'<svg viewBox="0 0 {W} {H + 26}" role="img" aria-label="Forecast against actual">'
            f'<line x1="0" x2="{W}" y1="{H}" y2="{H}" stroke="#C6D9E0"/>'
            f'<polygon points="{pts([*enumerate(hi), *reversed([*enumerate(lo)])])}" fill="#DCE9EF"/>'
            + line(p, "#3F6C8B") + line(act, "#0B1F3A")
            + "".join(f'<text x="{tx}" y="{H + 22}" text-anchor="{a}" font-size="17" fill="#4E6578">{t}</text>'
                      for tx, a, t in [(0, "start", "00:00"), (W / 2, "middle", "12:00"), (W, "end", "24:00")])
            + "</svg>")


def main(out: str) -> None:
    from playwright.sync_api import sync_playwright
    day = json.loads((ROOT / "site" / "data" / "days" / f"{ISSUE}.json").read_text())
    d = date.fromisoformat(day["delivery_date"])
    html = (PAGE.replace("__LABEL__", LABEL).replace("__DAY__", f"{d.day} {d:%B %Y}")
            .replace("__SVG__", chart(day["series"][SERIES], MODEL)))
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": 1200, "height": 630}, device_scale_factor=2)
        page.set_content(html, wait_until="networkidle")
        page.evaluate("document.fonts.ready")
        page.locator("#card").screenshot(path=out)
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1])
