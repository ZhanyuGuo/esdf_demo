import numpy as np

m, n = 3, 4  # 例如，3行4列
infinity_matrix = np.full_like(np.zeros((m, n)), np.inf)

print(infinity_matrix)