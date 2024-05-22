from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class SearchBase(ABC):
    dataset: [np.array, np.array]
    @abstractmethod
    def get_coef(self, dataset_):
        raise "Abstract method are invoked!"






