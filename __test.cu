#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel to add two vectors
__global__ void vectorAdd(int *A, int *B, int *C, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < N) {
        C[index] = A[index] + B[index]; // Add the elements
    }
}

int main() {
    int N = 1000; // Number of elements in the vectors
    int size = N * sizeof(int);

    // Allocate host memory
    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);
    int *h_C = (int *)malloc(size);

    // Initialize the vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Allocate device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Set up the execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            std::cout << "Error at index " << i << ": " << h_C[i]
                      << " != " << h_A[i] + h_B[i] << std::endl;
            return -1;
        }
    }

    std::cout << "Vector addition successful!" << std::endl;

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
