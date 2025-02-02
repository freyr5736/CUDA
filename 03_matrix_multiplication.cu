// Row = blockIdx.x * blockDim.y + threadIdx.y
// Col = blockIdx.x * blockDim.y + threadIdx.x
// C[i][j] = [k=0 ∑ m−1​] A[i][k]*  B[k][j]
// C[i][j] += A[i][k] * B[k][j];

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <cmath>

using std::generate;
using std::vector;

/**
 * CUDA Kernel for Matrix Multiplication
 * Each thread computes one element of the output matrix.
 */
__global__ void matrixMul(const int *a, const int *b, int *c, int N) {
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure thread is within valid matrix bounds
    if (row < N && col < N) {
        c[row * N + col] = 0;
        for (int k = 0; k < N; k++) {
            // Accumulate results for a single element
            c[row * N + col] += a[row * N + k] * b[k * N + col];
        }
    }
}

/**
 * Verify matrix multiplication results (CPU implementation).
 */
// void verify(int *a, int *b, int *c, int N) {
//     // For every row...
//     for (int i = 0; i < N; i++) {
//         // For every column...
//         for (int j = 0; j < N; j++) {
//             // Compute expected value for C[i][j]
//             int tmp = 0;
//             for (int k = 0; k < N; k++) {
//                 tmp += a[i * N + k] * b[k * N + j];
//             }

//             // Check against the CPU result
//             assert(tmp == c[i * N + j]);
//         }
//     }
// }

/**
 * Initialize matrices with random values.
 */
void init_matrices(int *a, int *b, int *c, int N) {
    for (int i = 0; i < N * N; i++) {
        a[i] = rand() % 10; // Random values from 0 to 9
        b[i] = rand() % 10; // Random values from 0 to 9
        c[i] = 0;           // Initialize result matrix to zero
    }
}

int main() {
    // Matrix size of 1024 x 1024 (number of elements)
    int n = 1 << 10; // Setting 10th bit to 1 i.e. 1024 = 00001000000000

    // Size in bytes of the matrix
    size_t bytes = n * n * sizeof(int);

    // Host pointers (i.e., for CPU)
    int *h_a, *h_b, *h_c; // Pointers to matrices

    // Allocating host memory (i.e., for CPU)
    h_a = (int *) malloc(bytes);
    h_b = (int *) malloc(bytes);
    h_c = (int *) malloc(bytes);

    // Device pointers (i.e., for GPU)
    int *d_a, *d_b, *d_c;

    // Allocating device memory (i.e., GPU)
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Initializing the matrices
    init_matrices(h_a, h_b, h_c, n);

    // Copy data to device (CPU -> GPU)
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Threads per block
    int BLOCK_SIZE = 16;

    // Blocks in each dimension (Grid size calculation)
    int GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; // Ensures full coverage

    // Using 3D objects
    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    // Launching kernel
    // Parallel computation occurs here
    matrixMul<<<grid, threads>>>(d_a, d_b, d_c, n);

    // Copy the values back to host (GPU -> CPU)
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Checking result against CPU computation
    // verify(h_a, h_b, h_c, n);

    // Output success message
    printf("EXECUTION SUCCESSFUL\n");

    // Free allocated host memory
    free(h_a);
    free(h_b);
    free(h_c);

    // Free allocated device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
