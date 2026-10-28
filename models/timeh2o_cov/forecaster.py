"""TimeH2O-v1 with future covariates, without test-time averaging (to fit in the live run). Env: timeh2o."""
from fl_timeh2o import run

run(covariates=True, fast=True)
