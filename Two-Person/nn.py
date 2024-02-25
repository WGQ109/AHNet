
import numpy as np

P_semantic = np.zeros((12,4))

P_semantic[[0,1,2],[0]] = 1
P_semantic[[3,4,5],[1]] = 1
P_semantic[[6,7,8],[2]] = 1
P_semantic[[9,10,11],[3]] = 1

# print(P_semantic)

P_symmetry = np.zeros((12,4))

P_symmetry[[0,1,2,9,10,11],[0]] = 1
P_symmetry[[3,8],[1]] = 1
P_symmetry[[4,7],[2]] = 1
P_symmetry[[5,6],[3]] = 1

# print(P_symmetry)