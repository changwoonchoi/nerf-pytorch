import numpy as np

z = np.zeros((20, 200, 200))
z = np.expand_dims(z, -1)

print(z.shape)