#!/usr/bin/env bash
# One Python 3.12 env per subprocess model family (CPU torch for the foundation models), built with uv.
# Run as root; safe to re-run.
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
  # classical statistical models through sktime's StatsForecast wrappers (statsforecast needs pandas 2,
  # the main venv has pandas 3); same sktime version as the main venv
  [stats]="sktime==1.2.0|statsforecast|pyyaml|joblib"
)
NO_TORCH=" stats "
families=("$@"); [ ${#families[@]} -eq 0 ] && families=("${!PKGS[@]}")
for f in "${families[@]}"; do
  (
    IFS='|' read -ra extra <<< "${PKGS[$f]}"
    [ -x "$E/$f/bin/python" ] || uv venv -q --python 3.12 "$E/$f"
    if [[ "$NO_TORCH" == *" $f "* ]]; then
      VIRTUAL_ENV="$E/$f" uv pip install -q pandas pyarrow "numpy<2.3" "${extra[@]}"
    else
      VIRTUAL_ENV="$E/$f" uv pip install -q "${CPU[@]}" torch pandas pyarrow "numpy<2.3" "${extra[@]}"
    fi
  ) > "$E/_logs/$f.log" 2>&1 &
done
wait
for f in "${families[@]}"; do
  mods="torch, pandas"; [[ "$NO_TORCH" == *" $f "* ]] && mods="pandas"
  if "$E/$f/bin/python" -c "import $mods" 2>/dev/null; then echo "  $f: ok"; else echo "  $f: FAILED, see $E/_logs/$f.log"; fi
done
chown -R foretec:foretec "$E" 2>/dev/null || true
mkdir -p /srv/foretec/.hf-cache && chown foretec:foretec /srv/foretec/.hf-cache
