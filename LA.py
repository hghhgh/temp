import numpy as np
data = [1,2,3]
l1 = np.linalg.norm(data,1)
l2 = np.linalg.norm(data)
print(l1, l2)