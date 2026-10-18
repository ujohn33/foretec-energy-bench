// Waiting-list form on compete.html: posts to /api/waitlist (foretec_live/waitlist.py) and shows the result in place.
// Without JavaScript the form still posts, and the server redirects back with ?joined=1 or ?error=...
(function () {
  const form = document.getElementById("waitlist-form");
  if (!form) return;
  const status = document.getElementById("w-status");
  const say = (msg, ok) => { status.textContent = msg; status.className = "status " + (ok ? "ok" : "err"); };
  const q = new URLSearchParams(location.search);
  if (q.get("joined")) say("Thank you, you are on the list.", true);
  else if (q.get("error")) say("That did not work: " + q.get("error"), false);

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const email = form.email.value.trim();
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return say("Please enter a valid email address.", false);
    if (!form.consent.checked) return say("Please tick the box so we may email you.", false);
    const btn = form.querySelector("button");
    btn.disabled = true;
    try {
      const r = await fetch("/api/waitlist", {
        method: "POST", headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ email, role: form.role.value, consent: true, website: form.website.value }),
      });
      const j = await r.json().catch(() => ({}));
      if (r.ok) { say("Thank you, you are on the list.", true); form.reset(); }
      else say(j.error || "Something went wrong, please try again later.", false);
    } catch (_) {
      say("Could not reach the server, please try again later.", false);
    } finally {
      btn.disabled = false;
    }
  });
})();
