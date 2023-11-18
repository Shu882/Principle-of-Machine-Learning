import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualization_savefig(mat, figname, title):
    fig, ax = plt.subplots()
    im = ax.imshow(mat)
    ax.set_axis_off()
    ax.set_title(title, usetex=True)
    plt.colorbar(im)
    fig.savefig(figname)
    plt.show()

# part (a)
three = pd.read_csv('three.txt', delimiter=' ', header=None)
three.drop(three.columns[256],axis=1, inplace=True)
line = 0
three_line1 = np.array(three.iloc[line, :])
visualization_savefig(three_line1.reshape([16, 16], order='F'), "fig1a_three.pdf", "First line in three.txt")

eight = pd.read_csv('eight.txt', delimiter=' ', header=None)
eight.drop(eight.columns[256],axis=1, inplace=True)
eight_line1 = np.array(eight.iloc[line, :])
visualization_savefig(eight_line1.reshape([16, 16], order='F'), "fig1b_eight.pdf", "First line in eight.txt")

# part (b)
x = np.vstack((np.array(three), np.array(eight)))
x_bar = x.mean(axis=0)
visualization_savefig(x_bar.reshape([16, 16], order='F'), "fig2_x_bar.pdf", r"$\bar{x}$")

# part (c)
# center X using x_bar and calculate the covariance matrix
x_centered = x - x_bar
n = x_centered.shape[0]
S = np.matmul(x_centered.transpose(), x_centered) / (n-1)
print("5x5 submatrix of the covariance matrix: \n%s" %np.round(S[:5, :5], 3))
# [[  59.16729323  142.14943609   28.68201754   -7.17857143  -14.3358396 ]
#  [ 142.14943609  878.93879073  374.13731203   24.12778195  -87.12781955]
#  [  28.68201754  374.13731203 1082.9058584   555.2268797    33.72431078]
#  [  -7.17857143   24.12778195  555.2268797  1181.24408521  777.77192982]
#  [ -14.3358396   -87.12781955   33.72431078  777.77192982 1429.95989975]]

# part (d)
w, v = np.linalg.eig(S)
ind = w.argsort()[-2:][::-1]

v1 = v[:, 0]
v2 = v[:, 1]
print("The two largest eigenvalues: \n%s" %np.round(w[ind], 3))
# [237155.246 145188.353]

visualization_savefig(v1.reshape((16, 16), order='F'), "fig4a_v1.pdf", "$v_1$")
visualization_savefig(v2.reshape((16, 16), order='F'), "fig4b_v2.pdf", "$v_2$")

# v1_cs = v1 - v1.min()
# v1_cs = v1_cs / v1_cs.max() * 255
# visualization_savefig(v1_cs.reshape((16, 16), order="F"), "testv1_2")

# part (e)
V = np.hstack((v1.reshape(-1, 1), v2.reshape(-1, 1)))
x_projected = np.matmul(x_centered, V)
coord_first_three = x_projected[0]
coord_first_eight = x_projected[200]
print("Resulting two coordinates for the first line in three.txt: \n%s" %np.round(coord_first_three, 3))
print("Resulting two coordinates for the first line in eight.txt: \n%s" %np.round(coord_first_eight, 3))
# [ 136.209 -242.628]
# [-312.687  649.573]

# part (f)
cumsum = 0
for i in range(n):
    ele = np.linalg.norm(np.matmul(x_centered[i, :], np.matmul(V, V.transpose())) - x_centered[i, :]) ** 2
    cumsum += ele
ave_rec_err = cumsum / n
print(f"Average reconstruction error: {np.round(ave_rec_err, 3)}")
# Average reconstruction error: 1405766.851

# part (g)
fig7, ax7 = plt.subplots()
ax7.scatter(x_projected[0:200, 0], x_projected[0:200, 1], color='red')
ax7.scatter(x_projected[200:400, 0], x_projected[200:400, 1], color='blue')
ax7.legend(("Three", "Eight"))
ax7.set_title("manual PCA")
ax7.set_xlabel('Principle Component 1')
ax7.set_ylabel('Principle Component 2')
fig7.savefig("fig7_2d_point_cloud.pdf")
fig7.show()

######## shorcut -- PCA in scikit-learn

pca = PCA(n_components=2)
pca.fit(x)
x_pca = pca.transform(x)

fig_pca, ax_pca = plt.subplots()
ax_pca.scatter(x_pca[0:200, 0], x_pca[0:200, 1], color='red', label='three')
ax_pca.scatter(x_pca[200:400, 0], x_pca[200:400, 1], color='blue', label='eight')
ax_pca.legend()
ax_pca.set_title("PCA in Scikit-learn")
ax_pca.set_xlabel('Principle Component 1')
ax_pca.set_ylabel('Principle Component 2')
fig_pca.savefig("fig8_2d_point_cloud_sklearn.pdf")
fig_pca.show()

# PCA tutorial: https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html#covariance-matrix