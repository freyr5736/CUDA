#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

// Verifying solution
void verify_result(float *a, float *b, float *c, int n) {
    float temp;
    float epsilon = 0.001; //since floating points can slightly be off we use epsilon to adjust for margin of error

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            temp = 0;
            for (int k = 0; k < n; ++k) {
                temp += a[k * n + i] * b[j * n + k];  //cuBLAS assume column major order. row major equivalent : a[i * n + k] * b[k * n + j];
            }
            assert(fabs(c[j * n + i] - temp) < epsilon);
        }
    }
}

int main(){
    //Initializing size
    int n = 1<<10;  //setting 10th bit to 1 (i.e 1024)
    size_t bytes = n*n*sizeof(float);


}