import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

obs = np.loadtxt(os.path.join('ovariancancer_obs.csv'), delimiter=',')

f = open(os.path.join('ovariancancer_grp.csv'), "r")
grp = f.read().split("\n")
print(obs)
U, S, VT = np.linalg.svd(obs, full_matrices=True)
print(U)
print(S)
print(VT)

plt.semilogy(S, '-o', color='k')
plt.grid(True)
plt.title('Singular Values')
plt.ylabel('Singular value')
plt.xlabel('k')
plt.savefig('output/pca1.png')
plt.show()
plt.plot(np.cumsum(S) / np.sum(S), '-o', color='k')
plt.grid(True)
plt.title('Singular Values: Cumulative Sum')
plt.ylabel('Cumulative energy')
plt.xlabel('k')
plt.savefig('output/pca2.png')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for j in range(obs.shape[0]):
    x = VT[0, :] @ obs[j, :].T
    y = VT[1, :] @ obs[j, :].T
    z = VT[2, :] @ obs[j, :].T

    if grp[j] == 'Cancer':
        cancer = ax.scatter(x, y, z, marker='x', color='r', s=50, label="Cancer")
    else:
        normal = ax.scatter(x, y, z, marker='o', color='b', s=50, label="Normal")

plt.legend(handles=[cancer, normal])
ax.view_init(25, 50)
ax.set_xlabel('First PC')
ax.set_ylabel('Second PC')
ax.set_zlabel('Third PC')

plt.savefig('output/3dpca.png')
plt.show()


plt.semilogy(S, '-o')
plt.title('Singular Values')
plt.ylabel('Singular value')
plt.xlabel('k')
plt.grid()
plt.savefig('output/pca_plot_1.png')
plt.show()


plt.plot(np.cumsum(S)/np.sum(S),'-o')
plt.title('Singular Values: Cumulative Sum')
plt.ylabel('Cumulative energy')
plt.xlabel('k')
plt.grid()
plt.savefig('output/pca_plot_2.png')
plt.show()