"""Tabby-Pretrain with covariates (ridge on the covariates, the backbone on the residual). Env: tabby."""
from fl_tabby import run

run("paris-noah/Tabby-Pretrain", covariates=True)
