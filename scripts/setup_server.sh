#!/usr/bin/env bash
# Set up Foretec Live on a fresh Ubuntu 24.04 server. Run as root from the unzipped repo:
#   sudo bash scripts/setup_server.sh            # core models only
#   sudo bash scripts/setup_server.sh --foundation  # also the foundation-model envs (~5 GB, see install_model_envs.sh)
#   sudo bash scripts/setup_server.sh --web      # also serve site/ on port 80 with Caddy
# Safe to re-run: it updates the code and venv in place and keeps data/ and results/.
set -euo pipefail

FOUNDATION=0; WEB=0
for a in "$@"; do
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
  echo "== web (Caddy on :80)"
  apt-get install -y -qq caddy >/dev/null
  cat > /etc/caddy/Caddyfile <<EOF
:80 {
  root * $APP/site
  file_server
}
EOF
  systemctl reload caddy || systemctl restart caddy
fi

cat <<EOF

Done. Next, as root:
  cd $APP && sudo -u foretec FORETEC_HOME=$APP foretec-live probe
  sudo -u foretec FORETEC_HOME=$APP foretec-live backfill --start 2026-09-01 --end 2026-09-28
  systemctl enable --now foretec-forecast.timer foretec-score.timer foretec-catchup.timer
  systemctl list-timers 'foretec*'
EOF
