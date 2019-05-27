#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <complex.h>

#define EQ_TOL 1e-8

typedef double complex* Matrix;

void matcpy (void * destmat, void * srcmat, size_t sz) 
{
  memcpy(destmat, srcmat, sz);
}

// Set all the values to zero in mat.
// mat is mxn
void zero_matrix(Matrix mat, int m, int n) {
    for (int i = 0; i < m*n; ++i) {
        mat[i] = 0;
    }
}

double scalar_inner_prod(double complex c) {
    return conj(c) * c;
}

// print a matrix
void pmat(Matrix A, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0;  j < n; ++j) {
            double complex c = A[i*n+j];
            printf("%3f + %3fi", creal(c), cimag(c));
            if (j == (n-1)) {
                printf("\n");
            } else {
                printf(", ");
            }
        }
    }
}

// calculate the dot product of column j_A in A and column j_B in B
// dimensions of two matrices both m by n
void columns_dot_product(double complex* result, Matrix A, Matrix B, int j_A, int j_B, int m, int n) {
    *result = 0;
    for(int i = 0; i < m; ++i) {
        *result += conj(A[i*n+j_A]) * B[i*n+j_B];
    }
}

void column_norm(double complex* sum, Matrix A, int j, int m, int n) {
    *sum = 0;
    for (int i = 0; i < m; ++i) {
        *sum += A[i*n+j];
    }
    *sum = cpow(*sum, 0.5);
}

void column_divide(double complex* factor, Matrix A, int j, int m, int n) {
    for(int i = 0; i < m; ++i) {
        A[i*n+j] /= *factor;
    }
}

// Implements the modified Gram Schmidt algorithm
// where A is mxn
//       R is nxn
//       Q overrides A
void ReducedQR(Matrix A, Matrix R, int m, int n) {

    for (int i = 0; i < n; ++i) {
        column_norm(R+(i*n+i), A, i, m, n);
        column_divide(R+(i*n+i), A, i, m, n);

        for(int j = i+1; j < n; ++j) {
            // A overridden by v and then by 
            columns_dot_product(R+(i*n+j), A, A, i, j, m, n);

            for (int k = 0; k < m; ++k) {
                A[k*n+j] -= R[i*n+j]*A[k*n+j];
            }
        }
    }
}

// Implements the householder method for orthogonal triangularisation
// A is mxn
// W is mxn
// W must be initialised to all zeros.
void house(Matrix A, Matrix W, int m, int n) {
    double sum_squares;
    double x_norm, v_norm;
    // factor by which x_norm gets multiplied to ensure numerical stability
    // (will be +- 1)
    double factor;

    // memory allocated for storing the rseults of dot product in inner loop
    // tempoarily
    Matrix dot_tmp = malloc(n * sizeof(double complex));

    for (int k = 0; k < n; ++k) {

        zero_matrix(dot_tmp, 1, n);
        x_norm = 0;
        v_norm = 0;
        // holds sum of squares of last m-k elements in x and v
        // used so we dont have to perform these operations twice
        sum_squares = 0;
        for (int i = k+1; i < m; ++i) {
            sum_squares += scalar_inner_prod(A[i*n+k]);
            W[i*n+k] = A[i*n+k];
        }

        // calculate x_norm and v_norm using the computed sum
        x_norm = sqrt(
                scalar_inner_prod(A[k*n+k])
                + sum_squares);

        // copy over the first element of v
        // not done in loop to avoid extra 'if' in the above loop
        /* W[k*n+k] = ( (creal(A[k*n+k]) >= 0)? -1 : 1) * A[k*n+k] + x_norm; */

        double complex diag = A[k*n+k];

        // handle edge case where entire vector gets zero'd by the naive algorithm
        if(fabs(sum_squares) < EQ_TOL
           && (x_norm - creal(diag)) < EQ_TOL
           && fabs(cimag(diag)) < EQ_TOL
           ) {
            if(fabs(((double) diag) - x_norm) < EQ_TOL) {
                factor = 1;
            } else {
                factor = -1;
            }
        } else if (creal(diag) >= 0) {
            factor = -1.;
            /* factor = 1.; */
        } else {
            factor = 1.;
        }
        W[k*n+k] = diag + factor * x_norm;

        v_norm = sqrt(scalar_inner_prod(W[k*n+k]) + sum_squares);
        printf("k: %d v_norm %3f x_norm %3f sum_squares %3f \n", k, v_norm, x_norm, sum_squares);

        for (int i = k; i<m; ++i) {
            W[i*n+k] /= v_norm;
        }

        // calculate the inner product
        // v_k* x A[k:m,k:n]
        for (int i = k; i < m; ++i) {
            for (int j = k; j < n; ++j) {
                dot_tmp[j-k] += conj(W[i*n+k]) * A[i*n+j];
            }
        }
        // perform subtraction
        // A[k:m, k:n] -= 2*v_k * inner product
        // effectively performing reflection thruogh the hyperspace
        for(int i = k; i < m; ++i) {
            for (int j = k; j < n; ++j) {
                A[i*n+j] -= 2*W[i*n+k]*dot_tmp[j-k];
            }
        }
    }
    free(dot_tmp);
}

int main() {
    int m, n;
    m = 3;
    n = 2;

    size_t mn_sz = m*n*sizeof(double complex);
    Matrix A = (Matrix) malloc(mn_sz);
    Matrix A_house = (Matrix) malloc(mn_sz);
    Matrix R = (Matrix) malloc(n * n * sizeof(double complex));

    A[0] = 1; A[1] = 0;
    A[2] = 0; A[3] = 1;
    A[4] = 1; A[5] = 0;
    matcpy(A_house, A, mn_sz);

    pmat(A, m, n);

    ReducedQR(A, R, m, n);
    printf("A_QR\n");
    pmat(A, m, n);
    printf("R\n");
    pmat(R, n, n);

    Matrix W = (Matrix) malloc(mn_sz);
    house(A_house, W, m, n);

    printf("W\n");
    pmat(W, n, n);
    printf("R_house\n");
    pmat(A_house, m, n);

    free(A);
    free(W);
    /* free(R); */
    return 0;
}
