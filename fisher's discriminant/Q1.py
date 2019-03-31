import numpy as np
import pandas as pd
from matplotlib import pyplot as plot
import numpy as np
import scipy.stats as stats
import math


def solve(m1, m2, std1, std2):
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
    return np.roots([a, b, c])


def calculate_covariance_matrix(X):
    # calculates the covariance matrix
    n_samples = X.shape[0]
    covariance_matrix = (X - X.mean(axis=0)).T.dot(X - X.mean(axis=0))

    return np.array(covariance_matrix, dtype=float)


class FisherDiscri():
    """Fisher's linear discriminant.
    """
    
    def __init__(self):
        self.w = [1, 1]
    
    # fitting the model to fisher lda
    def fit(self, X, y):

        # Separate data by class
        X1 = X[y == 0]
        X2 = X[y == 1]
        # Calculate the covariance matrices of the two datasets
        cov1 = calculate_covariance_matrix(X1)
        cov2 = calculate_covariance_matrix(X2)
        cov_tot = cov1 + cov2

        # Calculate the mean of the two datasets
        mean1 = X1.mean(0)
        mean2 = X2.mean(0)
        mean_diff = np.atleast_1d(mean1 - mean2)

        # Determine the vector which when X is projected onto it best separates the
        # data by class. w = (mean1 - mean2) / (cov1 + cov2)
        self.w = np.linalg.pinv(cov_tot).dot(mean_diff)

    # plotting the classes and their normal distribution
    def plotting(self, X, y):
        X1 = X[y == 0]
        X2 = X[y == 1]

        # calculating predicted values of x1 and x2
        predictionsx1 = X1.dot(self.w)
        plot.scatter(predictionsx1, np.zeros(500), c='b')

        predictionsx2 = X2.dot(self.w)
        plot.scatter(predictionsx2, np.zeros(500), c='r')

        # calculating mean, variance and standard deviation
        meanpredx1 = np.mean(predictionsx1)
        meanpredx2 = np.mean(predictionsx2)

        varpredx1 = np.var(predictionsx1)
        varpredx2 = np.var(predictionsx2)

        sigmax1 = math.sqrt(varpredx1)
        sigmax2 = math.sqrt(varpredx2)

        # determining normal curve of x1 and x2 and plotting them
        normalx1 = np.linspace(meanpredx1 - 3 * sigmax1, meanpredx1 + 3 * sigmax1, 500)
        plot.plot(normalx1, stats.norm.pdf(normalx1, meanpredx1, sigmax1))
        normalx2 = np.linspace(meanpredx2 - 3 * sigmax2, meanpredx2 + 3 * sigmax2, 500)
        plot.plot(normalx2, stats.norm.pdf(normalx2, meanpredx2, sigmax2))

        result = solve(meanpredx1, meanpredx2, sigmax1, sigmax2)
        plot.plot(result[1], stats.norm.pdf(result[1], meanpredx1, sigmax1), 'o')

        plot.savefig('plot_dataset_3' + '.png')
        plot.show()


classifier = FisherDiscri()
df = pd.read_csv('dataset_3.csv', header=None)
df.head()
df.drop(0, axis=1, inplace=True)

df = df.values

X = df[:, :-1]
y = df[:, -1]

classifier.fit(X, y)

classifier.plotting(X, y)
