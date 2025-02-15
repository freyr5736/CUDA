// Matrix multiplication using cuBLAS with unified memory and prefetching

#include <stdio.h>
#include <cassert>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using std::cout;

// Matrix dimensions (change as needed)
#define N 512  // Rows of A and C
#define M 512  // Columns of A and Rows of B
#define K 512  // Columns of B and C

// Main function
int main() {
    size_t bytes = N * M * sizeof(float);  // Size for A
    size_t bytesB = M * K * sizeof(float); // Size for B
    size_t bytesC = N * K * sizeof(float); // Size for C

    // Declare unified memory pointers
    float *A, *B, *C;

    // Allocate unified memory
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytesB);
    cudaMallocManaged(&C, bytesC);

    // Get device ID for prefetching
    int id;
    cudaGetDevice(&id);

    // Set memory hints and prefetch
    cudaMemAdvise(A, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(B, bytesB, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(C, bytesC, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);

    // Initialize matrices with random values
    for (int i = 0; i < N * M; i++) A[i] = static_cast<float>(rand() % 100);
    for (int i = 0; i < M * K; i++) B[i] = static_cast<float>(rand() % 100);

    // Prefetch to GPU
    cudaMemAdvise(A, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemAdvise(B, bytesB, cudaMemAdviseSetReadMostly, id);
    cudaMemPrefetchAsync(A, bytes, id);
    cudaMemPrefetchAsync(B, bytesB, id);
    cudaMemPrefetchAsync(C, bytesC, id);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Scalar values for cuBLAS (alpha, beta)
    const float alpha = 1.0f, beta = 0.0f;

    // Perform matrix multiplication using cuBLAS: C = A * B
    cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                   K, N, M,  // Output matrix (K x N), Input matrices (M x K) and (N x M)
                   &alpha, 
                   B, K,  // B (M x K) stored in column-major
                   A, M,  // A (N x M) stored in column-major
                   &beta, 
                   C, K);  // C (N x K) stored in column-major

    // Synchronize to ensure computation is complete
    cudaDeviceSynchronize();

    // Prefetch result back to CPU
    cudaMemPrefetchAsync(C, bytesC, cudaCpuDeviceId);

    // Verify results on CPU
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0.0f;
            for (int k = 0; k < M; k++) {
                sum += A[i * M + k] * B[k * K + j];
            }
            assert(fabs(C[i * K + j] - sum) < 1e-3);
        }
    }

    // Destroy cuBLAS handle
    cublasDestroy(handle);

    // Free unified memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    cout << "EXECUTION SUCCESSFUL!\n";

    return 0;
}
