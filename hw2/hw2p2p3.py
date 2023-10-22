from get_data import get_dataframe
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fit import (linfit, ridge, ridge2, get_mse)

url_mendota = "https://climatology.nelson.wisc.edu/first-order-station-climate-data/madison-climate/lake-ice/history-of-ice-freezing-and-thawing-on-lake-mendota/"
id_mendota = 'mendota'
df_mendota = get_dataframe(url=url_mendota, dataset=id_mendota, table_id=id_mendota)
print(df_mendota)

url_monona = "https://climatology.nelson.wisc.edu/first-order-station-climate-data/madison-climate/lake-ice/history-of-ice-freezing-and-thawing-on-lake-monona/"
id_monona = 'mendota'
df_monona = get_dataframe(url=url_monona, dataset='monona', table_id=id_monona)
print(df_monona)

# drop the unnecessary columns and merge
df_combo = df_mendota.merge(right=df_monona, how='inner', on='Winter')
print(df_combo)

df_combo.reset_index(inplace=True)
df_combo = df_combo.apply(pd.to_numeric, errors='coerce')

fig1, axes = plt.subplots(nrows=3, ncols=1, figsize=(10,18), sharex=True)
axes[0].scatter(df_combo.loc[:, 'Winter'], df_combo.loc[:, 'Days_monona'], c='r', marker='^', label="Monona")
axes[0].scatter(df_combo.loc[:, 'Winter'], df_combo.loc[:, 'Days_mendota'], c='g', marker='*', label="Mendota")

custom_y_ticks = [0, 50, 100, 150, 200]
axes[0].set_ylim(min(custom_y_ticks), max(custom_y_ticks))
axes[0].set_yticks(custom_y_ticks)
axes[0].legend()
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Ice Days')
axes[1].scatter(df_combo.loc[:, 'Winter'], df_combo.loc[:, 'Days_monona']-df_combo.loc[:, 'Days_mendota'], c='r', marker='^', label="Monona-Mendota")
axes[1].legend()
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Monona-Mendota Ice Days')
# fig1.show()

# (b)
train = df_combo[df_combo.iloc[:, 0] <= 1970]
test = df_combo[df_combo.iloc[:, 0] > 1970]
# print(train)
# print(test)

print(train.loc[:, 'Days_mendota'].mean())
print(train.loc[:, 'Days_monona'].mean())

print(train.loc[:, 'Days_mendota'].std())
print(train.loc[:, 'Days_monona'].std())

# 107.1896551724138
# 108.48275862068965
# 16.74666159754441
# 18.122521543826256

# (c)
trainx = np.array([train['Winter'], train['Days_monona']]).transpose()
trainy = np.array(train['Days_mendota'])
testx = np.array([test['Winter'], test['Days_monona']]).transpose()
testy = np.array(test['Days_mendota'])
coef = linfit(trainx, trainy)
print("\nCoeffients with OLS:\n", coef)
# Coeffients:
#  [-6.41827663e+01  4.12245664e-02  8.52950638e-01]

#(d)
ntrainy = len(trainy)
x = np.hstack((np.ones([ntrainy, 1]), trainx))
RS = (np.dot(x, coef) - trainy)**2
train_mse = RS.mean()
print(train_mse)
# 57.50963913639685
test_mse = get_mse(x=testx, y=testy, coef=coef)
print("Mean squared error on the test set:\n", test_mse)
# Mean squared error on the test set:
#  125.69670577805435
# (e)
coef2 = linfit(trainx[:, 0].reshape(-1, 1), trainy)
# print("\nCoeffients without Monona :\n", coef2)
# Coeffients without Monona :
#  [ 4.06111060e+02 -1.56298774e-01]
axes[2].scatter(trainx[:, 0], trainy, c='g', marker='o')
axes[2].set_xlabel('Year')
axes[2].set_ylabel('Mendota Ice Days')
axes[2].set_ylim(min(custom_y_ticks), max(custom_y_ticks))
axes[2].set_yticks(custom_y_ticks)
fitx = np.arange(start=trainx[:, 0].min(), stop=trainx[:, 0].max(), step=1)
fity = coef2[0] + coef2[1]* fitx
axes[2].plot(fitx, fity, linestyle='--', c='k', linewidth=4)
fig1.show()
fig1.savefig('hw2p2a_ice_days_pots.png')

# sign of gamma1 is neg.

#p3 (b)
# coef3 = ridge(trainx, trainy, lmd=0)
# print("\nCoeffients with ridge regression:\n", coef)
# Coeffients with ridge regression but tuning parameter 0: (this agrees with 2c! as it should be)
#  [-6.41827663e+01  4.12245664e-02  8.52950638e-01]

coef3 = ridge(trainx, trainy, lmd=1)
print("\nCoeffients with ridge regression:\n", coef3)
# Coeffients with ridge regression:
#  [-6.41667175e+01  4.12177636e-02  8.52922631e-01]

# (c)
lamds =  np.array([0.001,0.01,0.1,1.0,10])
nlamds = len(lamds)
ave_errs = np.zeros(nlamds)
nfold = 5
fold_size = len(trainx)//nfold

for i, lamd in enumerate(lamds):
    errs = np.zeros(nfold)
    for j in np.arange(nfold):
        start = j * fold_size
        end = (j + 1) * fold_size
        xtest = trainx[start:end]
        ytest = trainy[start:end]
        xtrain = np.concatenate([trainx[:start], trainx[end:]])
        ytrain = np.concatenate([trainy[:start], trainy[end:]])
        cv_coef = ridge(xtrain, ytrain, lmd=lamd)
        errs[j] = get_mse(xtest, ytest, cv_coef)
    ave_errs[i] = errs.mean()

fig2, ax2 = plt.subplots()
ax2.semilogx(lamds, ave_errs, marker='^', linestyle='--', markersize=10)
ax2.set_xlabel("Hyperparameter $\lambda$")
ax2.set_ylabel("MSE")
fig2.show()
fig2.savefig('p4c_learingCurve.png')

print(ave_errs)
# [82.27751414 82.27751839 82.27756091 82.27798631 82.28226759]
# conclusion: the smallest


fig2, axes1 = plt.subplots(nrows=3, ncols=1, figsize=(10, 18), sharex=True)
axes1[0].plot(df_combo.loc[:, 'Winter'], df_combo.loc[:, 'Days_monona'], c='r', label="Monona")
axes1[0].plot(df_combo.loc[:, 'Winter'], df_combo.loc[:, 'Days_mendota'], c='g', label="Mendota")

axes1[0].set_ylim(min(custom_y_ticks), max(custom_y_ticks))
axes1[0].set_yticks(custom_y_ticks)
axes1[0].legend()
axes1[0].set_xlabel('Year')
axes1[0].set_ylabel('Ice Days')
axes1[1].plot(df_combo.loc[:, 'Winter'], df_combo.loc[:, 'Days_monona']-df_combo.loc[:, 'Days_mendota'], c='r', label="Monona-Mendota")
axes1[1].legend()
axes1[1].set_xlabel('Year')
axes1[1].set_ylabel('Monona-Mendota Ice Days')

axes1[2].plot(trainx[:, 0], trainy, c='g', label='Mondota training')
axes1[2].set_xlabel('Year')
axes1[2].set_ylabel('Mendota Ice Days')
axes1[2].set_ylim(min(custom_y_ticks), max(custom_y_ticks))
axes1[2].set_yticks(custom_y_ticks)
axes1[2].plot(fitx, fity, linestyle='--', c='k', linewidth=4, label='Linear fit')
axes1[2].legend()
fig2.savefig("p2aplots.png")
fig2.show()


#(b) and (c) with ridge2
print("\n\n")
coef4 = ridge2(trainx, trainy, lmd=1)
print("\nCoeffients with ridge regression 2:\n", coef4)
# [-6.23294723e+01  4.04390872e-02  8.49714502e-01]

ave_errs2 = np.zeros(nlamds)
for i, lamd in enumerate(lamds):
    errs = np.zeros(nfold)
    for j in np.arange(nfold):
        start = j * fold_size
        end = (j + 1) * fold_size
        xtest = trainx[start:end]
        ytest = trainy[start:end]
        xtrain = np.concatenate([trainx[:start], trainx[end:]])
        ytrain = np.concatenate([trainy[:start], trainy[end:]])
        cv_coef = ridge2(xtrain, ytrain, lmd=lamd)
        errs[j] = get_mse(xtest, ytest, cv_coef)
    ave_errs2[i] = errs.mean()

fig3, ax3 = plt.subplots()
ax3.semilogx(lamds, ave_errs2, marker='^', linestyle='--', markersize=10)
ax3.set_xlabel("Hyperparameter $\lambda$")
ax3.set_ylabel("MSE")
fig3.show()
fig3.savefig('p4c_learingCurveridge2.png')

print(ave_errs2)
# [82.2775576  82.2779532  82.28193282 82.32406805 82.95984095]

