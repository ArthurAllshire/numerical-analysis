import numpy as np
import matplotlib.pyplot as plt


def approximate_legendre(n, samples=1000):
    x = np.linspace(-1., 1., samples).reshape(-1, 1)

    A = np.concatenate([x**p for p in range(n)], axis=1)

    Q, _ = np.linalg.qr(A)

    return x, Q


def plot_columns(x, mat):
    for i in range(mat.shape[1]):
        plt.plot(x, mat[:, i])
    plt.show()


x, legendre_matrix = approximate_legendre(4)
plot_columns(x, legendre_matrix)


# inner product should be approx 0
# high samples are discrete analogue of integral between -1, 1
# between two polynomials in L^2[-1, 1]
print(legendre_matrix[:, 0] @ legendre_matrix[:, 1])
