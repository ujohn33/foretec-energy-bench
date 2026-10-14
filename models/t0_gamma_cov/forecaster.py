"""t0-gamma with known-future covariates. Env: tfc06 (tfc-t0>=0.6.0)."""
from fl_t0 import run

run("theforecastingcompany/t0-gamma", covariates=True)
