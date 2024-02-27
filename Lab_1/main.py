import os
from math import gamma
import matplotlib.pyplot as plt
import pandas
import numpy as np
from IPython.display import display


# Normal distribution
def calc_normal(x, a, mu):
    return (1.0 / (a * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) ** 2) / (a ** 2))


# Cauchy distribution
def calc_cauchy(x, a, mu):
    return (1.0 / np.pi) * (a / ((x - mu) ** 2 + a ** 2))


# Student's distribution
def calc_student(x, t):
    return (gamma((t + 1.0) / 2.0) / (np.sqrt(t * np.pi) * gamma(t / 2.0))) * (
            (1.0 + (x ** 2) / t) ** (- (t + 1.0) / 2.0))


# Poisson distribution
def calc_poisson(x, *args):
    mu = args[0]
    return (mu ** x) * np.exp(-mu) / gamma(x + 1)


# Uniform distribution
def calc_uniform(x, a, b):
    return (1.0 / (b - a)) if (x >= a) and (x <= b) else 0.0


def print_graphics(N):
    # Normal distribution
    mu_normal = 0.0
    var_normal = 1.0
    sample_normal = [np.random.normal(mu_normal, var_normal, n) for n in N]

    # Cauchy distribution
    mu_cauchy = 0.0
    var_cauchy = 1.0
    sample_cauchy = [mu_cauchy + var_cauchy * np.random.standard_cauchy(n) for n in N]

    # Student's distribution
    t_student = 3.0
    sample_student = [np.random.standard_t(t_student, n) for n in N]

    # Poisson distribution
    mu_poisson = 10.0
    sample_poisson = [np.random.poisson(mu_poisson, n) for n in N]

    # Uniform distribution
    a_uniform = -np.sqrt(3)
    b_uniform = np.sqrt(3)
    sample_uniform = [np.random.uniform(a_uniform, b_uniform, n) for n in N]

    samples = [sample_normal, sample_cauchy, sample_student, sample_poisson, sample_uniform]

    densities = [lambda xi: calc_normal(xi, var_normal, mu_normal),
                 lambda xi: calc_cauchy(xi, var_cauchy, mu_cauchy),
                 lambda xi: calc_student(xi, t_student),
                 lambda xi: calc_poisson(xi, mu_poisson, mu_poisson),
                 lambda xi: calc_uniform(xi, a_uniform, b_uniform)]

    names = ['normal', 'cauchy', "student's", 'poisson', 'uniform']

    bins = [15, 25, 45]

    for j in range(len(samples)):
        plt.figure(figsize=(15, 5))
        plt.suptitle(f'Cumulative distribution function for {names[j]} distribution')
        for i in range(len(N)):
            plt.subplot(100 + len(N) * 10 + i + 1)
            x_min = min(samples[i][i])
            x_max = max(samples[i][i])
            x = np.linspace(x_min, x_max, 100)
            y = [densities[i](x_i) for x_i in x]
            plt.hist(samples[i][i], bins=bins[i], density=True, color='white', edgecolor='black')
            plt.plot(x, y, color='black', linewidth=1)
            plt.title(f'n = {N[i]}')
            plt.xlabel('values')
            plt.ylabel('CDF values')
            plt.yscale('linear')
            if N[i] > 500 and names[j] == 'cauchy':
                plt.yscale('log')
                plt.ylabel('log of CDF values')
        if not os.path.isdir("Lab_1/img/"):
            os.mkdir("Lab_1/img/")
        plt.savefig(f"Lab_1/img/{names[j]}.png", dpi=120)
        plt.close()


def print_characteristics(N):
    methods = [lambda n_: np.random.normal(0.0, 1.0, n_),
               lambda n_: np.random.standard_cauchy(n_),
               lambda n_: np.random.standard_t(3.0, n_),
               lambda n_: np.random.poisson(10.0, n_),
               lambda n_: np.random.uniform(-np.sqrt(3), np.sqrt(3), n_)]

    names = ['normal', 'cauchy', "student's", 'poisson', 'uniform']
    repeats = 1000

    for i in range(len(methods)):
        for n in N:
            data = np.zeros([2, 5])
            for j in range(repeats):
                sample = methods[i](n)

                sample.sort()
                x = np.mean(sample)
                med_x = np.median(sample)
                z_r = (sample[0] + sample[-1]) / 2.0
                z_q = (sample[int(np.ceil(n / 4.0) - 1)] + sample[int(np.ceil(3.0 * n / 4.0) - 1)]) / 2.0
                r = int(np.round(n / 4.0))
                z_tr = (1.0 / (n - 2 * r)) * sum([sample[i] for i in range(r, n - r)])

                stats = [x, med_x, z_r, z_q, z_tr]
                for k in range(len(stats)):
                    data[0][k] += stats[k]
                    data[1][k] += stats[k] * stats[k]

            data /= repeats
            data[1] -= data[0] ** 2
            df = pandas.DataFrame(data, columns=["x", "med x", "z_R", "z_Q", "z_{tr}"], index=["E(z)", "D(z)"])
            print(f"\n{names[i]} n = {n}")
            display(df)


N = [10, 100, 1000]
print_graphics(N)
print_characteristics(N)
