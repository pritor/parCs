
#include <stdio.h>
#include <malloc.h>
#include <math.h>
#define N 10000000




int main() {
    float *arr = (float*)malloc(sizeof(float)*N);		
    float sum = 0;		
    #pragma acc data create(arr[:N]) copy(sum)
    {
    #pragma acc parallel loop vector vector_length(32) gang	
    for(int i = 0; i<N;i++)
       arr[i] = sinf(2*M_PI*i/N);


    #pragma acc parallel loop reduction(+:sum)
    for(int i = 0;i<N;i++)
        sum+=arr[i];
    }
    printf("%le", sum);
    
    return 0;
}
