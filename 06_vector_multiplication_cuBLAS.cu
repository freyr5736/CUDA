#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

// Verify solution
// void verify_result(float *a, float *b, float *c, int n) {
//     float temp;
//     float epsilon = 0.001; // since floating points can slightly be off we use epsilon to adjust for margin of error

//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j < n; ++j) {
//             temp = 0;
//             for (int k = 0; k < n; ++k) {
//                 temp += a[k * n + i] *
//                         b[j * n +
//                           k]; // cuBLAS assume column major order. row major equivalent : a[i * n + k] * b[k * n +
//                           j];
//             }
//             assert(fabs(c[j * n + i] - temp) < epsilon);
//         }
//     }
// }

int main() {
    // Initialize size
    int n = 1 << 10; // setting 10th bit to 1 (i.e 1024)
    size_t bytes = n * n * sizeof(float);

    // Declare pointers to the matrices on device(gpu) and host(cpu)
    // float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    // Allocatmemory
    // h_a = (float *) malloc(bytes);
    // h_b = (float *) malloc(bytes);
    // h_c = (float *) malloc(bytes);
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Psuedo number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // Setting the seed
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // Fill the matrix with random numbers on the device(gpu)
    curandGenerateUniform(prng, d_a, n * n);
    curandGenerateUniform(prng, d_b, n * n);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Scaling factor
    float alpha = 1.0f;
    float beta = 0.0f;

    // Calculate c = (alpha*a) *b + (beta*c)
    //(m X n) * (n * k) = (m X k)
    // CUBLAS_OP_N convience functions
    // m,n,k are dimensions
    // lda = leading dimension of a
    // Signature (handle, operation, operation, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);

    // Copy back the three matrices
    // cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // // Verify solution
    // verify_result(h_a, h_b, h_c, n);

    printf("EXECUTION SUCCESSFUL");

    return 0;
}