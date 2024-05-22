import numpy as np
import pandas as pd

from COUSWORK.searchers.Likelihood import Ll
from COUSWORK.searchers.MNK import MNK
from COUSWORK.searchers.Rabas import Rob

SOLVERS = {"Like": Ll,
           "Mnk": MNK,
           "Rabas": Rob}


def generate():
    np.random.seed(0)
    N = 100
    a_real = 2.5
    b_real = 0.9
    x = np.random.rand(N)
    y = a_real + b_real * x + 0.1 * np.random.randn(N)
    return [x, y]


def execute():
    data = generate()
    buff = []
    for solver in SOLVERS:
        buff.append(SOLVERS[solver].get_coef(data))
    return buff


def create_report(data_report):
    for report in data_report:
        print(pd.DataFrame(report))


create_report(execute())
