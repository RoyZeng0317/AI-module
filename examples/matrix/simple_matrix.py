import sympy as sp
A=sp.Matrix([[1,0,-1],[0,2,-1],[-1,1,0]])
A.det(), A.inv(), A.row_join(sp.eye(3)).rref()