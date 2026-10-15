"""Tabby with covariates (ridge on the covariates, Tabby on the residual). Env: tabby."""
from fl_tabby import run

run("paris-noah/Tabby", covariates=True)
