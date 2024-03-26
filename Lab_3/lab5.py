import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import math


def lab5():
    def multivariate_normal(mean1, mean2, std1, std2, rho, n):
        means = np.array([mean1, mean2])
        covs = np.array([
            [std1 ** 2, std1 * std2 * rho],
            [std1 * std2 * rho, std2 ** 2],
        ])
        return np.random.default_rng().multivariate_normal(means, covs, n).T

    ns = np.array([20, 60, 100])
    rhos = np.array([20, 0.5, 0.9])

    for n in ns:
        for rho in rhos:
            pearson_coefs = np.array([])
            spearman_coefs = np.array([])
            quadrant_coefs = np.array([])

            for i in range(1000):
                x, y = multivariate_normal(0, 0, 1, 1, rho, n)
                pearson_coef = pearsonr(x, y)[0]
                spearman_coef = spearmanr(x, y)[0]
                quadrant_coef = math.sqrt(2) * np.mean(np.sign(x - np.median(x)) * np.sign(y - np.median(y)))

                pearson_coefs = np.append(pearson_coefs, pearson_coef)
                spearman_coefs = np.append(spearman_coefs, spearman_coef)
                quadrant_coefs = np.append(quadrant_coefs, quadrant_coef)

            print(f'\nSample size: {n}, Correlation coefficient (rho): {rho}')
            print('Pearson coefficients: Mean =', np.mean(pearson_coefs), 'Mean of squares =',
                  np.mean(pearson_coefs ** 2), 'Variance =', np.var(pearson_coefs))
            print('Spearman coefficients: Mean =', np.mean(spearman_coefs), 'Mean of squares =',
                  np.mean(spearman_coefs ** 2), 'Variance =', np.var(spearman_coefs))
            print('Quadrant coefficients: Mean =', np.mean(quadrant_coefs), 'Mean of squares =',
                  np.mean(quadrant_coefs ** 2), 'Variance =', np.var(quadrant_coefs))

    for n in ns:
        pearson_coefs = np.array([])
        spearman_coefs = np.array([])
        quadrant_coefs = np.array([])

        for i in range(1000):
            x, y = multivariate_normal(0, 0, 1, 1, 0.9, n) * 0.9 + multivariate_normal(0, 0, 10, 10, -0.9,
                                                                                       n) * 0.1
            pearson_coef = pearsonr(x, y)[0]
            spearman_coef = spearmanr(x, y)[0]
            quadrant_coef = math.sqrt(2) * np.mean(np.sign(x - np.median(x)) * np.sign(y - np.median(y)))

            pearson_coefs = np.append(pearson_coefs, pearson_coef)
            spearman_coefs = np.append(spearman_coefs, spearman_coef)
            quadrant_coefs = np.append(quadrant_coefs, quadrant_coef)

        print(f'\nSample size: {n}')
        print('Pearson coefficients: Mean =', np.mean(pearson_coefs), 'Mean of squares =',
              np.mean(pearson_coefs ** 2),
              'Variance =', np.var(pearson_coefs))
        print('Spearman coefficients: Mean =', np.mean(spearman_coefs), 'Mean of squares =',
              np.mean(spearman_coefs ** 2), 'Variance =', np.var(spearman_coefs))
        print('Quadrant coefficients: Mean =', np.mean(quadrant_coefs), 'Mean of squares =',
              np.mean(quadrant_coefs ** 2), 'Variance =', np.var(quadrant_coefs))

        def add_confidence_ellipse(x, y, ax, std, **kwargs):
            cov = np.cov(x, y)
            pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
            ell_radius_x = np.sqrt(1 + pearson)
            ell_radius_y = np.sqrt(1 - pearson)
            ellipse = Ellipse(
                (0, 0),
                width=ell_radius_x * 2,
                height=ell_radius_y * 2,
                facecolor='none',
                **kwargs,
            )

            scale_x = np.sqrt(cov[0, 0]) * std
            mean_x = np.mean(x)

            scale_y = np.sqrt(cov[1, 1]) * std
            mean_y = np.mean(y)

            transf = transforms.Affine2D() \
                .rotate_deg(45) \
                .scale(scale_x, scale_y) \
                .translate(mean_x, mean_y)

            ellipse.set_transform(transf + ax.transData)
            return ax.add_patch(ellipse)

        x, y = multivariate_normal(0, 0, 1, 1, 0.9, 1000) * 0.9 + multivariate_normal(0, 0, 10, 10, -0.9,
                                                                                      1000) * 0.1

    fig, ax = plt.subplots(1, 1)
    ax.scatter(x, y, s=0.5)

    add_confidence_ellipse(x, y, ax, 1, edgecolor='red')
    add_confidence_ellipse(x, y, ax, 2, edgecolor='fuchsia', linestyle='--')
    add_confidence_ellipse(x, y, ax, 3, edgecolor='blue', linestyle=':')



    plt.show()
