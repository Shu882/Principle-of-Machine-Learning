import numpy as np
from scipy.stats import t
from scipy import stats

A = np.array([1, 2, 1, 2, 4, 3, 2, 3, 4, 1, 2, 1])
B = np.array([1, 4, 3, 1, 5, 5, 4, 4, 2, 3, 4, 2])
nA = len(A)
nB = len(B)

#(a)
alpha = 0.05

LA = A.mean() - t.ppf(1-alpha/2, df=nA-1) * np.std(A, ddof=1) / np.sqrt(nA)
UA = A.mean() + t.ppf(1-alpha/2, df=nA-1) * np.std(A, ddof=1) / np.sqrt(nA)


LB = B.mean() - t.ppf(1-alpha/2, df=nB-1) * np.std(B, ddof=1) / np.sqrt(nB)
UB = B.mean() + t.ppf(1-alpha/2, df=nB-1) * np.std(B, ddof=1) / np.sqrt(nB)
print(LA.round(3), UA.round(3), '\n')
print(LB.round(3), UB.round(3), '\n')

# 1.458 2.875
# 2.27 4.058

#(b)
diff = A-B
se_diff = np.std(diff, ddof=1)
tscore = diff.mean()/(se_diff/np.sqrt(len(diff)))
# tscore
p_value = 2*t.cdf(tscore, df=11)
print("p value is: \n", p_value)
# p value is:
#  0.026094682241525755

# double check with paired t test function from scipy
results = stats.ttest_rel(A, B)
print(results.pvalue)
