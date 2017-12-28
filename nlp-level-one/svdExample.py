# Creation date 27th Dec 2017.

# This is an example of SVD implementation. It shows how we can construct approximation of the input matrix.
# Note that We should use python's SVD API rather than use scipy/numpy's linalg. The reason being that eigen-vector
# aren't unique, oftentimes, it will result worse approximation. R's eigenvalue and eigenvector calculations will provide better approximations.
# Please refer to svdExample.R for comparison.

import numpy as np 

from sklearn.decomposition import TruncatedSVD

from numpy import linalg as LA

# SVD formula: A = U x S x V^t

# Input.
A = np.matrix([[1,1,1, 0, 0], [3,3,3, 0, 0],[4,4, 4, 0, 0], [5,5,5,0,0],[0,2,0,4,4],[0,0,0,5,5],[0,1,0,2,2]])

# Variance and Covariance for U, 
Usquare = A*np.transpose(A)

# Variance and Covariance for V, 
Vsquare = np.transpose(A) * A

# Derive U (eigenvector of SU).
U = LA.eig(Usquare)[1]

# Derive S from SU.
SU = np.sqrt(LA.eig(Usquare)[0])

# Derive V (eigenvector of SV).
V = LA.eig(Vsquare)[1]

# Derive S from SV.
SV = np.sqrt(LA.eig(Vsquare)[0])


n = len(SU)
eigenValueMatrix = np.zeros((n, n))

for i in range(n):
	eigenValueMatrix[i, i] = SU[i]

# From SU, we can see that only the first 3 eigenvalues are positive, and the rest are either nan or 0. therefore we will set k = 3
eigenValueMatrix = eigenValueMatrix[:3, :3]

U = U[:, :3] 

Aapprox = U * eigenValueMatrix * (V.T[:3, :])

# We can see that the Aapprox is approximate equal to A, however, we have reduce our dimension from 7 to 3.

model = TruncatedSVD(n_components=3)

a = model.fit_transform(A)

model.explained_variance_ratio_  

model.components_ # note that model.components_ should be similar to V.T[:3, :].



# --- scipy

# from scipy.linalg import eig

# w, vl, vr = eig(Usquare, left=True)

# The End.




