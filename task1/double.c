#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <malloc.h>
#include <math.h>
#define N 10000000




int main() {
    double *arr = (double*)malloc(sizeof(double)*N);		
    double sum = 0;
    cublasHandle_t handle;
    cublasCreate(&handle);    
    #pragma acc data create(arr[:N]) copy(sum)
    {
    #pragma acc parallel loop vector vector_length(32) gang	
    for(int i = 0; i<N;i++)
       arr[i] = sin(2*M_PI*i/N);
	

    }
    cublasDasum(handle, N*8, arr,1,  &sum);
    printf("%le", sum);
    cublasDestroy(handle);
    return 0;
}
