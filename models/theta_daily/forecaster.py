from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.transformations.series.detrend import Deseasonalizer


def build(target, zone):
    return TransformedTargetForecaster([
        ("deseason", Deseasonalizer(model="additive", sp=96)),
        ("theta", ThetaForecaster(deseasonalize=False)),
    ])
