import numpy as np
from scipy.optimize import minimize
from COUSWORK.searchers.Search import SearchBase


class LeastSquares(SearchBase):
    def get_coef(self, dataset_):
        x, y = dataset_

        xy_m = np.mean(np.multiply(x, y))
        x_m = np.mean(x)
        x_2_m = np.mean(np.multiply(x, x))
        y_m = np.mean(y)

        beta_hat = (xy_m - x_m * y_m) / (x_2_m - x_m ** 2)
        alpha_hat = y_m - x_m * beta_hat

        y_pred = alpha_hat + beta_hat * x
        residuals = y - y_pred
        alpha_er = np.sqrt(np.sum(residuals ** 2) / (len(x) - 2)) * np.sqrt(
            1 / len(x) + x_m ** 2 / np.sum((x - x_m) ** 2))
        beta_er = np.sqrt(np.sum(residuals ** 2) / (len(x) - 2)) * np.sqrt(1 / np.sum((x - x_m) ** 2))

        return {"alpha_hat": [alpha_hat],
                "beta_hat": [beta_hat],
                "alpha_er": [alpha_er],
                "beta_er": [beta_er]}


MNK = LeastSquares()