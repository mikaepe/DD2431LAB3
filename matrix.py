import numpy as np


# DIMENSIONS
rows = 7
cols = 4
depth = 5

#CREATE MATRIX
A = np.zeros((rows,cols))
print A.shape

A = np.zeros((rows,cols))
print A
A = np.ones((rows,cols))
print A
A = np.array([(1,2,5),(5,5,1),(1,1,1)])
print A
d = np.diag(A)
print d
B = np.diag(d)
print B

AA = (1.0/A)
print AA
print AA*A

print np.linalg.inv(A)
