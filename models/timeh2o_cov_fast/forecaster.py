"""TimeH2O-v1 with future covariates, no test-time averaging. Env: timeh2o."""
from fl_timeh2o import run

run(covariates=True, fast=True)
