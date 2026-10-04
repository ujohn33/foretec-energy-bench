#!/usr/bin/env bash
# Set up Foretec Energy Bench on a fresh Ubuntu 24.04 server. Run as root from the unzipped repo:
#   sudo bash scripts/setup_server.sh            # core models only
#   sudo bash scripts/setup_server.sh --foundation  # also the foundation-model envs (~5 GB, see install_model_envs.sh)
#   sudo bash scripts/setup_server.sh --web      # also serve site/ with Caddy (plain HTTP on :80)
#   sudo bash scripts/setup_server.sh --domain=energybench.foretec.co --alias=transparency.foretec.co
#       HTTPS on that domain, the aliases redirect to it; both remembered for later runs
# Safe to re-run: it updates the code and venv in place and keeps data/ and results/.
set -euo pipefail

FOUNDATION=0; WEB=0; DOMAIN=; ALIASES=
for a in "$@"; do
  case "$a" in --domain=*) DOMAIN="${a#--domain=}"; WEB=1; continue ;; esac
  case "$a" in --alias=*) ALIASES="${a#--alias=}"; WEB=1; continue ;; esac
  case "$a" in
    --foundation|--chronos) FOUNDATION=1 ;;
    --web) WEB=1 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done

[ "$(id -u)" -eq 0 ] || { echo "run as root (sudo)"; exit 1; }
SRC="$(cd "$(dirname "$0")/.." && pwd)"
BASE=/srv/foretec
APP=$BASE/foretec-live
VENV=$BASE/venv

echo "== packages"
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip rsync tzdata libgomp1 >/dev/null

echo "== user and folders"
id foretec >/dev/null 2>&1 || useradd --system --home-dir $BASE --create-home --shell /usr/sbin/nologin foretec
mkdir -p $APP
chmod 755 $BASE
rsync -a --delete \
  --exclude /data/ --exclude /results/ --exclude /site/ --exclude /geo/raw/ --exclude '*.egg-info' --exclude __pycache__ --exclude .venv \
  "$SRC"/ $APP/
mkdir -p $APP/data $APP/results $APP/site
touch $BASE/foretec.env && chmod 600 $BASE/foretec.env   # put ENTSOE_API_KEY=... here if you switch source to entsoe
chown -R foretec:foretec $BASE

echo "== python environment"
sudo -u foretec python3 -m venv $VENV
sudo -u foretec $VENV/bin/pip install -q --upgrade pip
sudo -u foretec $VENV/bin/pip install -q -e "$APP"
if [ $FOUNDATION -eq 1 ]; then
  echo "== foundation-model envs"
  bash $APP/scripts/install_model_envs.sh
fi
ln -sf $VENV/bin/foretec-live /usr/local/bin/foretec-live

echo "== systemd timers"
cp $APP/scripts/systemd/*.service $APP/scripts/systemd/*.timer /etc/systemd/system/
systemctl daemon-reload
# once the daily timers are on (after the first probe + backfill), keep the full set enabled on every re-run
if systemctl is-enabled -q foretec-forecast.timer 2>/dev/null; then
  systemctl enable --now foretec-forecast.timer foretec-score.timer foretec-catchup.timer
fi
# timers are installed but NOT enabled: run the probe and a backfill first, then:
#   systemctl enable --now foretec-forecast.timer foretec-score.timer foretec-catchup.timer

if [ $WEB -eq 1 ]; then
  apt-get install -y -qq caddy >/dev/null
  # pages and data revalidate on every load (app.js/style.css are cache-busted by name, vendor/ never changes)
  FRESH=$(printf '  @fresh path / *.html /data/*\n  header @fresh Cache-Control "no-cache"\n')
  FRESH="$FRESH"$'\n'
  # the domain and aliases are remembered, so a later plain --web never downgrades HTTPS back to plain HTTP
  [ -n "$DOMAIN" ] && echo "$DOMAIN" > $BASE/site_domain
  [ -z "$DOMAIN" ] && [ -f $BASE/site_domain ] && DOMAIN=$(cat $BASE/site_domain)
  [ -n "$ALIASES" ] && echo "$ALIASES" > $BASE/site_aliases
  [ -z "$ALIASES" ] && [ -f $BASE/site_aliases ] && ALIASES=$(cat $BASE/site_aliases)
  if [ -n "$DOMAIN" ]; then
    IP=$(curl -4 -s --max-time 10 https://api.ipify.org || true)
    # only names whose DNS already points here: Caddy would otherwise keep failing certificate orders for them
    points_here() { [ -z "$IP" ] || getent ahostsv4 "$1" | awk '{print $1}' | grep -qx "$IP"; }
    LIVE=(); for h in ${ALIASES//,/ }; do points_here "$h" && LIVE+=("$h") || echo "   $h: no DNS record to $IP yet, skipped"; done
    if points_here "$DOMAIN"; then
      SERVE="$DOMAIN"; REDIR=("${LIVE[@]}")
    else
      # main domain not in DNS yet: keep serving on the aliases that are, re-run --web once it is
      echo "   $DOMAIN: no DNS record to $IP yet; serving on ${LIVE[*]:-nothing} until it has one"
      SERVE=$(IFS=,; echo "${LIVE[*]}"); REDIR=()
    fi
    [ -n "$SERVE" ] || { echo "no configured name points at this server"; exit 1; }
    TARGET=${SERVE%%,*}
    echo "== web (Caddy, https://$TARGET${REDIR:+; ${REDIR[*]} redirect there}; plain http on the server IP too)"
    {
      printf '%s {\n  root * %s/site\n  encode gzip\n%s  file_server\n}\n' "${SERVE//,/, }" "$APP" "$FRESH"
      for h in "${REDIR[@]}"; do printf '\n%s {\n  redir https://%s{uri} permanent\n}\n' "$h" "$TARGET"; done
      if [ -n "$IP" ]; then printf '\nhttp://%s {\n  redir https://%s{uri} permanent\n}\n' "$IP" "$TARGET"; fi
    } > /etc/caddy/Caddyfile
  else
    echo "== web (Caddy on :80, no domain: pass --domain=example.org for HTTPS)"
    printf ':80 {\n  root * %s/site\n%s  file_server\n}\n' "$APP" "$FRESH" > /etc/caddy/Caddyfile
  fi
  caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null
  systemctl reload caddy || systemctl restart caddy
fi

cat <<EOF

Done. Next, as root:
  cd $APP && sudo -u foretec FORETEC_HOME=$APP foretec-live probe
  sudo -u foretec FORETEC_HOME=$APP foretec-live backfill --start 2026-09-01 --end 2026-09-28
  systemctl enable --now foretec-forecast.timer foretec-score.timer foretec-catchup.timer
  systemctl list-timers 'foretec*'
EOF
