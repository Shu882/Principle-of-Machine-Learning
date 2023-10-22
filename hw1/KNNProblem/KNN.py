import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

def KNNPredict(D, K, xtest):
    # D training set (with both x and y), K # of nearest neighbors used, x test point
    x = D[:,:-1]
    y = D[:,-1]
    N = x.shape[0] # number of samples and features
    S = np.empty([N,3]) # store distance to training example n
    S = pd.DataFrame(S, columns = ['TrainDataPoints', 'Distances', 'Labels'])
    distances = np.sqrt(np.sum((x - xtest)**2, axis=1))
    S.loc[:,'TrainDataPoints'] = range(N)
    S.loc[:,'Distances'] = distances
    S.loc[:,'Labels'] = y
    S.sort_values(by='Distances', inplace=True)
    yhat = S['Labels'].iloc[:K].mean()
    return np.sign(yhat)

# def KNNErrorRate():

############################
# function test
""""
# Generate synthetic KNN training data

X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=2, random_state=42)

# Visualize the data
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', marker='o', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', marker='x', label='Class 1')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('KNN Training Data')
plt.legend()
# plt.show()

# Display the first few rows of the generated data
print("Generated Data (first 5 rows):\n", X[:5])
print("Labels (first 5):\n", y[:5])

# util
y[y==0] = -1
D = np.column_stack((X, y.reshape(-1, 1)))
label = KNNPredict(D,3,np.array([1,0]))
print("Prediction: ", label)

"""
