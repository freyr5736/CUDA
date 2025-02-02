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

using std::cout;
using std::generate;
using std::vector;

// Pull out matrix and shared memory tile size
const int N = 1 << 10;
const int SHMEM_SIZE = 1 << 10; // Shared memory size. 10th bit is 1 (i.e 1024)

__global__ void matrixMul(const int *a, const int *b, int *c) {
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Statically allocated shared memory
    __shared__ int s_a[SHMEM_SIZE];
    __shared__ int s_b[SHMEM_SIZE];

    // Accumulate in temporary variable
    int tmp = 0;

    // Sweep tile across matrix
    for (int i = 0; i < N; i += blockDim.x) {
        // Load in elements for this tile

        // Accesses an element in a 2D matrix stored in a 1D array (row-major order)
        // row * N         : Moves to the start of the correct row
        // i + threadIdx.x : Selects the specific column within that row
        // Example: If N = 4, row = 2, i = 1, threadIdx.x = 2, this accesses Matrix[2][3]
        s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x];

        // Accesses an element in a 2D matrix stored in a 1D array (row-major order)
        // i * N           : Moves 'i' rows down in the matrix
        // threadIdx.y * N : Further shifts by threadIdx.y rows (parallel access in y-dimension)
        // col             : Moves to the correct column
        // Example: If N = 4, i = 1, threadIdx.y = 2, col = 3, this accesses b[3][3]
        s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + col];

        // Wait for both tiles to be loaded in before doing computation
        __syncthreads();

        // Do matrix multiplication on the small matrix
        for (int j = 0; j < blockDim.x; j++) {
            tmp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
        }

        // Wait for all threads to finish using current tiles before loading in new ones
        __syncthreads();
    }

    // Write back results
    c[row * N + col] = tmp;
}

// Check result on the CPU
// void verify_result(vector<int> &a, vector<int> &b, vector<int> &c) {
//     // For every row...
//     for (int i = 0; i < N; i++) {
//         // For every column...
//         for (int j = 0; j < N; j++) {
//             // For every element in the row-column pair
//             int tmp = 0;
//             for (int k = 0; k < N; k++) {
//                 // Accumulate the partial results
//                 tmp += a[i * N + k] * b[k * N + j];
//             }

//             // Check against the CPU result
//             assert(tmp == c[i * N + j]);
//         }
//     }
// }

int main() {
    // Size (in bytes) of matrix
    size_t bytes = N * N * sizeof(int);

    // Host vectors
    vector<int> h_a(N * N);
    vector<int> h_b(N * N);
    vector<int> h_c(N * N);

    // Initialize matrices
    generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
    generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data to the device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // Threads per CTA dimension
    int THREADS = 32;

    // Blocks per grid dimension (assumes THREADS divides N evenly)
    int BLOCKS = N / THREADS;

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    // Launch kernel
    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);

    // Copy back to the host
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Check result
    // verify_result(h_a, h_b, h_c);

    cout << "EXECUTION SUCCESSFULLY\n";

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}