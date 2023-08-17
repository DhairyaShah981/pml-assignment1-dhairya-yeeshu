import numpy as np
y = [1, 2, 3, 4, 5, 6, 7, 8, 9]
q_hat = np.quantile(y, 0.1)
# mathematically
# i = 0.1(9-1) = 0.8
# j = 0.8 - 0 = 0.8
# q_hat = y[0] + 0.8(y[1] - y[0]) = 1 + 0.8(2-1) = 1.8
print(q_hat)

# how does np.quantile work?
# https://numpy.org/doc/stable/reference/generated/numpy.quantile.html