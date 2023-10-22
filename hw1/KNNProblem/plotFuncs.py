import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plotLearningCurve(M, errorRates, plotTitle, plotFilename, color='red'):
    plt.figure(figsize=(8, 6))
    plt.xlabel('Training Examples')
    plt.ylabel('Error Rate')
    plt.title(plotTitle)
    plt.plot(M, errorRates, c=color)
    # plt.legend(loc='upper right')
    plt.savefig(plotFilename)
    # plt.show()


def plotTuneK(K, errorRates, plotTitle, plotFilename, label, color='red'):
    plt.figure(figsize=(8, 6))
    plt.plot(K, errorRates, c=color, label=label)
    plt.xlabel('k')
    plt.ylabel('Error Rate')
    plt.title(plotTitle)
    plt.legend(loc='upper right')
    plt.savefig(plotFilename)
    # plt.show()


def plotTuneMaxDepth(max_depths, errorRates, plotTitle, plotFilename, label, color='red'):
    plt.figure(figsize=(8, 6))
    plt.plot(max_depths, errorRates, c=color, label=label)
    plt.xlabel('max_depth')
    plt.ylabel('Error Rate')
    plt.title(plotTitle)
    plt.legend(loc='upper right')
    plt.savefig(plotFilename)
    # plt.show()


## function tests
""" 
# tests have passed

data = pd.read_csv("errorRates_S_CV.csv")
data = np.array(data)
plotTuneK(data[:,0], data[:,1], "testTuneK", "testTuneK.png", 'test', 'green')

data = pd.read_csv("errorRates_sklearn_CV.csv")
data = np.array(data)
plotTuneMaxDepth(data[:,0], data[:,2], "testTuneK", "testTuneK.png", 'test', 'green')

data = pd.read_csv("learningcurve_S.csv")
data = np.array(data)
plotLearningCurve(data[:,0], data[:,1], "testTuneK", "testTuneK.png", 'green')
"""


