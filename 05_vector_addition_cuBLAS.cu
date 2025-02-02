#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <iostream>
// Initializing the vector
void vector_init(float *a, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = (float) (rand() % 100);
    }
}

// Verify the result
void verify_result(float *a, float *b, float *c, float factor, int n) {
    for (int i = 0; i < n; ++i) {
        assert(c[i] == factor * a[i] + b[i]); // Ensure computed result is correct
    }
}

int main() {
    // Vector size
    int n = 1 << 2; // second bit set to 1 (i.e 2^2 = 4)
    size_t bytes = n * sizeof(float);

    // Declaring vector pointers
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b;

    // Allocating memory
    h_a = (float *) malloc(bytes);
    h_b = (float *) malloc(bytes);
    h_c = (float *) malloc(bytes);
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);

    // Initialize vectors
    vector_init(h_a, n);
    vector_init(h_b, n);

    // Creating and initializing a new context
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Copying the vectors over to the device (i.e GPU)
    cublasSetVector(n, sizeof(float), h_a, 1, d_a, 1);
    cublasSetVector(n, sizeof(float), h_b, 1, d_b, 1);

    // Launching simple SAXPY kernel (single precision a*x + y)
    const float scale = 2.0f;
    cublasSaxpy(handle, n, &scale, d_a, 1, d_b, 1);

    // Copying result vector back to host (i.e CPU)
    cublasGetVector(n, sizeof(float), d_b, 1, h_c, 1);

    // Verifying the result
    verify_result(h_a, h_b, h_c, scale, n);

    // Cleaning up the created handle
    cublasDestroy(handle);

    // Releasing the allocated memory
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);
    free(h_c);

    std::cout << "EXECUTION SUCCESSFUL\n";
    return 0;
}
