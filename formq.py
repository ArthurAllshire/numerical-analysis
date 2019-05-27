import numpy as np


def formQ(W):
    """Forms Q from matrix W resulting from householder triangularisation."""
    m = W.shape[0]
    n = W.shape[1]
    Q = np.identity(m)
    # bad and inefficent code - should be written to take advantage of sparsity
    for k in range(n):
        Q_k = np.identity(m)
        w_i = W[k:, k]
        Q_k[k:, k:] -= 2 * np.outer(w_i, w_i)
        Q = Q.dot(Q_k)
    return Q


if __name__ == "__main__":
    W = np.array([[-0.382683, 0.],
                  [0., 1.],
                  [0.923880, 0.]])
    print(formQ(W))
