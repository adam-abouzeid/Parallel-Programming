#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openacc.h>


double* randomMatrix(int rows, int cols) {
    double* matrix = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)rand() / RAND_MAX * 100;  // Random double between 0 and 100
    }
    return matrix;
}


void multiplyMatrices(int m, int n, int p, double* A, double* B, double* C) {
    #pragma acc data copyin(A[0:m*n], B[0:n*p]) copyout(C[0:m*p])
    {
        #pragma acc parallel
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += A[i * n + k] * B[k * p + j];
                }
                C[i * p + j] = sum;
            }
        }
    }
}

int main() {
    srand(time(NULL));

 int m = 1000, n = 600, p = 800; // Example: A is 500x300, B is 300x400

    clock_t start, end;
    double cpu_time_used, total_time = 0;

    for (int iteration = 0; iteration < 10; iteration++) {
        double* A = randomMatrix(m, n);
        double* B = randomMatrix(n, p);
        double* C = (double*)malloc(m * p * sizeof(double));  // Result matrix

        start = clock();
        multiplyMatrices(m, n, p, A, B, C);
        end = clock();

        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        total_time += cpu_time_used;

        free(A);
        free(B);
        free(C);
    }

    printf("Average time taken over 10 iterations: %.6f seconds\n", total_time / 10);
    return 0;
}
