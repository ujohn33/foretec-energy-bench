from lightgbm import LGBMRegressor
from sktime.forecasting.compose import make_reduction


def build(target, zone):
    reg = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1)
    return make_reduction(reg, window_length=192, strategy="recursive")
