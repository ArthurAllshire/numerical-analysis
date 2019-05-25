import numpy as np
import matplotlib.pyplot as plt


def legendre_analytic_order_3(x):
    return [1, x, 0.5*(3*x**2 - 1), 0.5*(5*x**3-3*x)]


def approximate_legendre(n, samples=257):
    x = np.linspace(-1., 1., samples).reshape(-1, 1)

    A = np.concatenate([x**p for p in range(n+1)], axis=1)

    Q, _ = np.linalg.qr(A)

    # legendre polynomials satisfy P_k(1) = 1
    Q = Q @ np.diag(1 / Q[-1, :])

    return x, Q


def plot_columns(x, mat):
    for i in range(mat.shape[1]):
        plt.plot(x, mat[:, i])
    plt.show()


def legendre_errors(x, mat):
    analytical = []
    for x_ in x.reshape(-1):
        analytical.append(legendre_analytic_order_3(x_))
    print(analytical[:5])
    diff = np.array(analytical) - mat
    return diff


if __name__ == "__main__":
    x, legendre_qr = approximate_legendre(3)
    print(legendre_qr[:5])
    errors = legendre_errors(x, legendre_qr)
    plot_columns(x, errors)

    # inner product should be approx 0
    # high samples are discrete analogue of integral between -1, 1
    # between two polynomials in L^2[-1, 1]
    print(legendre_qr[:, 0] @ legendre_qr[:, 1])
