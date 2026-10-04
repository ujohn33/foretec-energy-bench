#!/usr/bin/env bash
# One Python 3.12 env per foundation-model family, CPU torch, built with uv. Run as root; safe to re-run.
#   bash scripts/install_model_envs.sh [family ...]     (default: all)
set -u
E=${FORETEC_ENVS:-/srv/foretec/envs}
export UV_CACHE_DIR=/srv/foretec/.uv-cache UV_PYTHON_INSTALL_DIR=/srv/foretec/.uv-python
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
mkdir -p "$E/_logs"
CPU=(--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match)
declare -A PKGS=(
  [chronos]="chronos-forecasting>=2.0"
  [timesfm]="timesfm[torch,xreg] @ git+https://github.com/google-research/timesfm.git"
  [moirai]="uni2ts @ git+https://github.com/SalesforceAIResearch/uni2ts.git"
  [tirex]="tirex-ts"
  [toto]="toto-ts|setuptools<81"
  [sundial]="transformers==4.40.1"
  [ttm]="granite-tsfm"
  [tfc]="tfc-t0>=0.5.0"
)
families=("$@"); [ ${#families[@]} -eq 0 ] && families=("${!PKGS[@]}")
for f in "${families[@]}"; do
  (
    IFS='|' read -ra extra <<< "${PKGS[$f]}"
    [ -x "$E/$f/bin/python" ] || uv venv -q --python 3.12 "$E/$f"
    VIRTUAL_ENV="$E/$f" uv pip install -q "${CPU[@]}" torch pandas pyarrow "numpy<2.3" "${extra[@]}"
  ) > "$E/_logs/$f.log" 2>&1 &
done
wait
for f in "${families[@]}"; do
  if "$E/$f/bin/python" -c "import torch, pandas" 2>/dev/null; then echo "  $f: ok"; else echo "  $f: FAILED, see $E/_logs/$f.log"; fi
done
chown -R foretec:foretec "$E" 2>/dev/null || true
mkdir -p /srv/foretec/.hf-cache && chown foretec:foretec /srv/foretec/.hf-cache
