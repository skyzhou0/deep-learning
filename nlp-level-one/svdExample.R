A = matrix(c(1,1,1, 0, 0, 3,3,3, 0, 0,4,4, 4, 0, 0, 5,5,5,0,0,0,2,0,4,4,0,0,0,5,5,0,1,0,2,2), ncol = 7, nrow = 5)

A = t(A)

Usquare = A %*% t(A)

Vsquare = t(A) %*% A

S = sqrt(eigen(Usquare)$values)

U = (eigen(Usquare)$vectors)

V= (eigen(Vsquare)$vectors)

# making a diagnal eigenvalue matrix.

Smatrix =  diag(S)

# set k = 3
U = U[,1:3]
Smatrix = Smatrix[1:3, 1:3]

Aapprox = U %*% Smatrix %*% (t(V)[1:3, ])

# result by choose k = 3, we have perfect approximation of the input matrix A.