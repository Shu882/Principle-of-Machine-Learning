import numpy as np

# b) function to draw samples
def draw(size, seed=None):
    nvar = 2
    # x = np.random.uniform(size=[size, nvar])
    rng = np.random.default_rng(seed=seed)
    x = rng.uniform(size=[size, nvar])
    conditions = [x < 1/3, (x >= 1/3) & (x < 2/3), x >= 2/3]
    replacement_values = [1, 2, 3]
    xtransformed = np.select(conditions, replacement_values, default=x)
    index = xtransformed - 1
    index = index.astype(np.int64)
    A = np.array([[.1, .2, .2],[.2, .4, .8], [.2, .8, .9]])
    y = np.empty(size)
    # for row in index.astype(np.int64):
    for i in range(size):
        p = A[index[i,0], index[i,1]]
        # y[i] = np.random.binomial(size=1,n=1,p=p)
        y[i] = np.random.choice([0, 1], p=[1 - p, p])
        y[y<1] = -1
        # y = y.astype(np.int64)
    return np.column_stack((x, y.reshape(-1, 1)))

######################
## function test
"""
sample = draw(30)
print(sample)
"""


