import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from COUSWORK.searchers.Search import SearchBase


class Likelihood(SearchBase):
    def get_coef(self, dataset_):
        def neg_log_likelihood(params):
            alpha, beta, sd = params
            y_pred = alpha + beta * np.linspace(0, 1, 100)
            negative_log_likelihood = -np.sum(np.log(norm.pdf(self.dataset[1], loc=y_pred, scale=sd)))
            return negative_log_likelihood

        self.dataset = dataset_
        result = minimize(neg_log_likelihood,
                          np.array([1, 1, 1]),
                          method='nelder-mead')
        alpha_hat, beta_hat, sd_hat = result.x
        standard_errors = [0, 0]
        # hessian_matrix = result.hess_inv
        # standard_errors = np.sqrt(np.diag(hessian_matrix))

        return {"alpha_hat": [alpha_hat],
                "beta_hat": [beta_hat],
                "alpha_er": [standard_errors[0]],
                "beta_er": [standard_errors[1]]}


Ll = Likelihood()
