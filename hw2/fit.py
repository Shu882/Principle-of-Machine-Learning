import numpy as np
import statsmodels as sm


def linfit(x, y):
    """
    both x and y are numpy arrays
    """
    n = len(y)
    x = np.hstack((np.ones([n, 1]), x))
    x_transpose = x.transpose()
    x_transpose_x = np.dot(x_transpose, x)
    x_transpose_y = np.dot(x_transpose, y)
    x_transpose_x_inv = np.linalg.inv(x_transpose_x)
    coef = np.dot(x_transpose_x_inv, x_transpose_y)
    return coef


def ridge(x, y, lmd=0):
    """
    x: feature matrix
    y: target
    l: penalzing parameter
    """
    n = len(y)
    x = np.hstack((np.ones([n, 1]), x))
    p = x.shape[1]
    A = np.eye(p)
    A[0, 0] = 0
    x_transpose = x.transpose()
    x_transpose_x = np.dot(x_transpose, x)
    x_transpose_y = np.dot(x_transpose, y)
    combo_inv = np.linalg.inv(x_transpose_x + lmd*A)
    coef = np.dot(combo_inv, x_transpose_y)
    return coef

def ridge2(x, y, lmd=0):
    """
    x: feature matrix
    y: target
    l: penalzing parameter
    slightly different since the n term will be multiplied too!
    """
    n = len(y)
    x = np.hstack((np.ones([n, 1]), x))
    p = x.shape[1]
    A = np.eye(p)
    A[0, 0] = 0
    x_transpose = x.transpose()
    x_transpose_x = np.dot(x_transpose, x)
    x_transpose_y = np.dot(x_transpose, y)
    combo_inv = np.linalg.inv(x_transpose_x + n*lmd*A)
    coef = np.dot(combo_inv, x_transpose_y)
    return coef

def get_mse(x, y, coef):
    n = len(y)
    x = np.hstack((np.ones([n, 1]), x))
    RS = (np.dot(x, coef) - y) ** 2
    mse = RS.mean()
    return mse


# linear fitting and ridge regression test
x1 = np.linspace(start=0, stop=4, num=5)
x2 = x1**2
y = 100 + 21*x1 + 13*x2 + np.random.default_rng(seed=2023).normal(size=len(x1))
x = np.stack([x1, x2], axis=1)
# print(linfit(x, y))
print(ridge(x, y, lmd=0))


