#include "stdlib.h"
#include "stdio.h"
/* #include "math.h" */
#include "complex.h"

typedef double complex* Matrix;

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

// where A is mxn
//       R is nxn
//       Q overrides A
void ReducedQR(Matrix A, Matrix R, int m, int n) {
    // todo: override A in algorithm
    /* A[0] = m; */
    /* R[0] = n; */
    
    for (int i = 0; i < n; ++i) {
        column_norm(R+(i*n+i), A, i, m, n);
        column_divide(R+(i*n+i), A, i, m, n);

        for(int j = i+1; j < n; ++j) {
            // A overridden by v and then by 
            printf("%d\n", j);
            columns_dot_product(R+(i*n+j), A, A, i, j, m, n);

            for (int k = 0; k < m; ++k) {
                A[k*n+j] -= R[i*n+j]*A[k*n+j];
            }
            printf("done\n");
        }
    }
}

int main() {
    int m, n;
    m = 3;
    n = 2;

    Matrix A = (Matrix) malloc(m * n * sizeof(double complex));
    Matrix R = (Matrix) malloc(n * n * sizeof(double complex));

    A[0] = 1; A[1] = 0;
    A[2] = 0; A[3] = 1;
    A[4] = 1; A[5] = 0;
    pmat(A, m, n);


    ReducedQR(A, R, m, n);

    printf("R\n");
    pmat(R, n, n);
    printf("A\n");
    pmat(A, m, n);

    free(A);
    free(R);
    return 0;
}
