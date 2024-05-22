import numpy as np
from scipy.optimize import minimize

class RobustEstimator:

    def get_coef(self, dataset_):
        def abs_dev_val(b_arr, x, y):
            return np.sum(np.abs(y - b_arr[0] - b_arr[1] * x))

        x_, y_ = dataset_
        init_b = np.array([0, 1])
        res = minimize(abs_dev_val, init_b, args=(x_, y_), method='SLSQP')

        return {"alpha_hat": [res.x[0]],
                "beta_hat": [res.x[1]],
                "alpha_er": [0],
                "beta_er": [0]}


Rob = RobustEstimator()