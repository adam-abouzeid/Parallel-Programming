#include <stdio.h>
#include <stdlib.h>
#include <time.h>


double** randomMatrix(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX * 100;  // Random double between 0 and 100
        }
    }
    return matrix;
}


void multiplyMatrices(int m, int n, int p, double** A, double** B, double** C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i][j] = 0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}


void freeMatrix(int rows, double** matrix) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int main() {
    srand(time(NULL));  

   
    int m = 1000, n = 600, p = 800; //dimensions of matrices

   
    clock_t start, end;
    double cpu_time_used, total_time = 0;


    for (int iteration = 0; iteration < 10; iteration++) {
        double** A = randomMatrix(m, n);
        double** B = randomMatrix(n, p);

      
        double** C = randomMatrix(m, p); // Allocating C filling not necessry

        start = clock();
        multiplyMatrices(m, n, p, A, B, C);
        end = clock();

        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        total_time += cpu_time_used;

     
        freeMatrix(m, A);
        freeMatrix(n, B);
        freeMatrix(m, C);
    }

    printf("Average time taken over 10 iterations: %.6f seconds\n", total_time / 10);

    return 0;
}
