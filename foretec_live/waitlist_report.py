"""Email report of the competition waiting list (see waitlist.py), sent through the Resend API.

Runs once a day (scripts/systemd/foretec-waitlist-report.timer). It emails when there are new sign-ups since the last
report, and every Monday in any case, so a silent week still shows the job is alive. Standard library only.

  env (/etc/foretec-waitlist.env, root only)  RESEND_API_KEY, REPORT_TO (your address), REPORT_FROM (optional; without
                                              a verified domain Resend sends from onboarding@resend.dev, and then only
                                              to the address the Resend account was opened with)
  reads    /var/lib/foretec-waitlist/waitlist.jsonl
  state    $STATE_DIRECTORY/reported  (how many entries the last report covered)

python3 waitlist_report.py           daily run
python3 waitlist_report.py --test    send a report now, whatever the count
python3 waitlist_report.py --dry     print the email instead of sending it
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

LIST = Path(os.environ.get("WAITLIST_FILE", "/var/lib/foretec-waitlist/waitlist.jsonl"))
STATE = Path(os.environ.get("STATE_DIRECTORY", "/var/lib/foretec-waitlist-report")) / "reported"
ROLE = {"participant": "participant", "sponsor": "sponsor", "researcher": "researcher", "curious": "just curious"}


def main(argv: list[str]) -> int:
    rows = [json.loads(x) for x in LIST.read_text().splitlines() if x.strip()] if LIST.exists() else []
    done = int(STATE.read_text()) if STATE.exists() else 0
    new = rows[done:]
    monday = dt.date.today().weekday() == 0
    if not new and not monday and "--test" not in argv and "--dry" not in argv:
        return 0
    roles = collections.Counter(r["role"] for r in rows)
    subject = f"Competition waiting list: {len(new)} new, {len(rows)} in total"
    lines = [f"{len(new)} new sign-up{'s' if len(new) != 1 else ''} since the last report, {len(rows)} in total.", ""]
    if new:
        lines += ["New:"] + [f"  {r['t'][:16].replace('T', ' ')} UTC  {ROLE.get(r['role'], r['role']):<12}  {r['email']}" for r in new] + [""]
    if rows:
        lines += ["All sign-ups by role:"] + [f"  {ROLE.get(k, k):<12}  {v}" for k, v in roles.most_common()] + [""]
    lines += ["Full list on the server:  ssh transparency 'cat /var/lib/foretec-waitlist/waitlist.jsonl'",
              "Page: https://energybench.foretec.co/compete.html"]
    body = "\n".join(lines)
    if "--dry" in argv:
        print(f"Subject: {subject}\n\n{body}")
        return 0
    key, to = os.environ.get("RESEND_API_KEY"), os.environ.get("REPORT_TO")
    if not key or not to:
        print("RESEND_API_KEY and REPORT_TO must be set in /etc/foretec-waitlist.env", file=sys.stderr)
        return 1
    msg = {"from": os.environ.get("REPORT_FROM") or "Energy Bench <onboarding@resend.dev>", "to": [to],
           "subject": subject, "text": body}
    req = urllib.request.Request("https://api.resend.com/emails", data=json.dumps(msg).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "User-Agent": "foretec-energy-bench/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
    except urllib.error.HTTPError as e:
        print(f"Resend refused the report: HTTP {e.code} {e.read().decode()[:300]}", file=sys.stderr)
        return 1
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(str(len(rows)))
    print(f"report sent: {len(new)} new, {len(rows)} total")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
