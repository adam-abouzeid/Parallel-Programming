#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16  // Define the size of the tile for tiling in the kernel


__global__ void matrixMultiply(double *A, double *B, double *C, int m, int n, int p) {
   
    __shared__ double tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ double tile_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    
    for (int t = 0; t < (n + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
      
        int idxA = row * n + t * TILE_WIDTH + threadIdx.x;
        if (idxA / n == row && idxA % n < n) {  
            tile_A[threadIdx.y][threadIdx.x] = A[idxA];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0;
        }

        int idxB = (t * TILE_WIDTH + threadIdx.y) * p + col;
        if (idxB / p == t * TILE_WIDTH + threadIdx.y && idxB % p == col) {  // Check within matrix boundaries
            tile_B[threadIdx.y][threadIdx.x] = B[idxB];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();  
     
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();  
    }


    if (row < m && col < p) {
        C[row * p + col] = sum;
    }
}


double* randomMatrix(int rows, int cols) {
    double* matrix = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)rand() / RAND_MAX * 100;  // Random double between 0 and 100
    }
    return matrix;
}

int main() {
    srand(time(NULL));
    int m = 500, n = 300, p = 400;  // Dimensions of the matrices

    size_t sizeA = m * n * sizeof(double);
    size_t sizeB = n * p * sizeof(double);
    size_t sizeC = m * p * sizeof(double);

    double *A = randomMatrix(m, n);
    double *B = randomMatrix(n, p);
    double *C = (double*)malloc(sizeC);

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((p + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);

    matrixMultiply<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, p);

    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}
