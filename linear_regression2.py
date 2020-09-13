import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size': 20})

# Load dataset
H, y = load_boston(return_X_y=True)
b = H[:,-1] # housing values in $1000s
A = H[:,:-1] # other factors

# Pad with ones for nonzero offset
A = np.pad(A,[(0,0),(0,1)],mode='constant',constant_values=1)

n = 400
btrain = b[1:n]
Atrain = A[1:n]
btest = b[n:]
Atest = A[n:]

# Solve Ax=b using SVD
U, S, VT = np.linalg.svd(Atrain,full_matrices=0)
x = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ btrain


plt.plot(btrain, Color='k', LineWidth=2, label='Housing Value') # True relationship
plt.plot(Atrain@x, '-o', Color='r', LineWidth=1.5, MarkerSize=6, label='Regression')
plt.xlabel('Neighborhood')
plt.ylabel('Median Home Value [$1k]')
plt.legend()
plt.savefig('output/regressionris.png')
plt.show()

# sort_ind = np.argsort(H[:,n])
# btest = btest[sort_ind] # sorted values
plt.plot(btest, Color='k', LineWidth=2, label='Housing Value') # True relationship
plt.plot(Atest@x, '-o', Color='r', LineWidth=1.5, MarkerSize=6, label='Regression')
plt.xlabel('Neighborhood')
plt.legend()
plt.savefig('output/regression2.png')
plt.show()


