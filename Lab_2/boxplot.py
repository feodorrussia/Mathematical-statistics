import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


def normal_distribution(n):
    return np.random.normal(0, 1, n)


def cauchy_distribution(n):
    return np.random.standard_cauchy(n)


def student_distribution(n):
    return np.random.standard_t(3, n)


def poisson_distribution(n):
    return np.random.poisson(10, n)


def uniform_distribution(n):
    return np.random.uniform(-math.sqrt(3), math.sqrt(3), n)


def print_boxplot():
    distributions = [
        ('normal', normal_distribution),
        ('cauchy', cauchy_distribution),
        ('student', student_distribution),
        ('poisson', poisson_distribution),
        ('uniform', uniform_distribution),
    ]
    ns = np.array([20, 100])

    for distribution_name, distribution_f in distributions:
        figure, axes = plt.subplots(2, 1, figsize=(20, 10))

        for index, n in enumerate(ns):
            values = distribution_f(n)

            sns.boxplot(values, ax=axes[index], orient='h', color='skyblue', medianprops=dict(color='orange'),
                        whiskerprops=dict(color='green'),
                        flierprops=dict(marker='o', markersize=5, markerfacecolor='red'))

            axes[index].set_xlabel('x')
            axes[index].set_title(f'n = {n}')

        figure.savefig(f'plots/{distribution_name}.png')
