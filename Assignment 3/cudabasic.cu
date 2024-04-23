#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define BLOCK_SIZE 16
__global__ void matrixMultiply(double *A, double *B, double *C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

double* randomMatrix(int size) {
    double* matrix = (double*)malloc(size * sizeof(double));
    for (int i = 0; i < size; i++) {
        matrix[i] = (double)rand() / RAND_MAX * 100;
    }
    return matrix;
}

int main() {
    srand(time(NULL));
    int m = 500, n = 300, p = 400;

    size_t sizeA = m * n * sizeof(double);
    size_t sizeB = n * p * sizeof(double);
    size_t sizeC = m * p * sizeof(double);

    double *A = randomMatrix(m * n);
    double *B = randomMatrix(n * p);
    double *C = (double*)malloc(sizeC);

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((p + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0, totalSeconds = 0;

    for (int i = 0; i < 10; i++) {
        cudaEventRecord(start);
        matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, p);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&milliseconds, start, stop);
        totalSeconds += milliseconds / 1000.0;  // Convert milliseconds to seconds
    }

    printf("Average time taken over 10 iterations: %.6f seconds\n", totalSeconds / 10);

    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
