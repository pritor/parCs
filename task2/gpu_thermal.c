#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

int main(int argc, char** argv) {
    double accuracy;
    int size, iternum;
    accuracy = atof(argv[1]);
    size = atoi(argv[2]);
    iternum = atoi(argv[3]);

    
    double* array = (double*)calloc(size * size, sizeof(double));
    double* arraynew = (double*)calloc(size * size, sizeof(double));

    array[0] = 10.0;
    array[size-1] = 20.0;
    array[(size-1) *size] = 20.0;
    array[size * size-1] = 30.0;

    arraynew[0] = 10.0;
    arraynew[size - 1] = 20.0;
    arraynew[(size - 1) * size] = 20.0;
    arraynew[size * size - 1] = 30.0;

    double error = 1.0;
    double step = 10.0/(size-1);
    int realsize = size*size;
#pragma acc parallel loop
    for (int i = 1; i < size; i++) {
        array[i] = array[0] + step * i;
        array[size * (size - 1) + i] = array[(size - 1) * size] + step*i;
        array[(size * i)] = array[0] + step*i;
        array[size - 1 + i * size] = array[size - 1] + step * i;
    }

    int k = 0;
    memcpy(arraynew, array, size * size * sizeof(double));
    clock_t begin = clock();
#pragma acc enter data copyin(array[0:realsize], arraynew[0:realsize])
    { 
        for (; (k < iternum) && (error > accuracy); k++) {
            error =0;

#pragma acc data present(array, arraynew)   
#pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(128) reduction(max:error)	    
            for (int i = 1; i < size - 1; i++) {
                for (int j = 1; j < size - 1; j++) {
                    arraynew[j + i * (size)] = 0.25 * (array[j + (i + 1) * (size)] + array[j + (i - 1) * (size)] + array[j - 1 + i * (size)] + array[j + 1 + i * (size)]);
                    error = fmax(error, (arraynew[j + i * (size)] - array[j + i * (size)]));
                }

            }
	    
	    double* temp = array;
            array = arraynew;
            arraynew = temp;
        
    
	}}
//#pragma acc exit data copyout(error, k)    
    clock_t end= clock();
    printf("%d %lf\n", k, error);
    printf("time: %le\n",1.0 * (end-begin)/CLOCKS_PER_SEC);
    free(array);
    free(arraynew);
	

    return 0;
}
