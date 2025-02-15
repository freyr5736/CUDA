// This program computes the sum of two N-element vectors using cuBLAS and unified memory prefetch

#include <stdio.h>
#include <cassert>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using std::cout;

// Main function
int main() {
    // Array size of 2^16 (65536 elements)
    const int N = 1 << 16;
    size_t bytes = N * sizeof(float);

    // Declare unified memory pointers
    float *a, *b, *c;

    // Allocate unified memory
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Get the device ID for prefetching calls
    int id;
    cudaGetDevice(&id);

    // Set memory hints and prefetch
    cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemPrefetchAsync(c, bytes, id);

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        a[i] = static_cast<float>(rand() % 100);
        b[i] = static_cast<float>(rand() % 100);
    }

    // Prefetch 'a' and 'b' arrays to the GPU
    cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemPrefetchAsync(a, bytes, id);
    cudaMemPrefetchAsync(b, bytes, id);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Scalar for SAXPY operation
    const float alpha = 1.0f;

    // Copy 'b' to 'c' before performing cuBLAS operation
    cudaMemcpy(c, b, bytes, cudaMemcpyDeviceToDevice);

    // Perform vector addition using cuBLAS: c = a + b (by modifying c)
    cublasSaxpy_v2(handle, N, &alpha, a, 1, c, 1);

    // Synchronize device to ensure computation is finished
    cudaDeviceSynchronize();

    // Verify the result on the CPU
    for (int i = 0; i < N; i++) {
        assert(fabs(c[i] - (a[i] + b[i])) < 1e-5);
    }

    // Destroy cuBLAS handle
    cublasDestroy(handle);

    // Free unified memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    cout << "EXECUTION SUCCESSFUL!\n";

    return 0;
}
