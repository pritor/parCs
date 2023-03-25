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
    array[(size-1) *size] = 20.0;                                                 //угловые значения
    array[size * size-1] = 30.0;

    double error = 1.0;
    double step = 10.0/(size-1);
    int realsize = size*size;
#pragma acc parallel loop
    for (int i = 1; i < size; i++) {
        array[i] = array[0] + step * i;                                           //значения рамки
        array[size * (size - 1) + i] = array[(size - 1) * size] + step*i;
        array[(size * i)] = array[0] + step*i;
        array[size - 1 + i * size] = array[size - 1] + step * i;
    }

    int k = 0;
    memcpy(arraynew, array, size * size * sizeof(double));
#pragma acc enter data copyin(array[0:realsize], arraynew[0:realsize], error)
    { 
    	clock_t begin = clock();
        for (; (k <= iternum) && (error > accuracy); k++) {
	    		
	    if(k%100==0){
		
		                                            //обнуление ошибки и перенос её на gpu каждые 100 итераций   
            	error =0;
		
	       	#pragma acc update device(error)  
		  
		}
#pragma acc data present(array, arraynew, error)   
#pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(128) reduction(max:error)	    //основной алгоритм
            for (int i = 1; i < size - 1; i++) {
                for (int j = 1; j < size - 1; j++) {
                    arraynew[j + i * (size)] = 0.25 * (array[j + (i + 1) * (size)] + array[j + (i - 1) * (size)] + array[j - 1 + i * (size)] + array[j + 1 + i * (size)]);
                    error = fmax(error, (arraynew[j + i * (size)] - array[j + i * (size)]));
                }

            }
            

	    if(k%100==0){
		#pragma acc update host(error)                           //обновление ошибки на cpu   
	    }
	    double* temp = array;
            array = arraynew;
            arraynew = temp;
        
    
	}
    
    clock_t end= clock();
    printf("time: %le\n",1.0 * (end-begin)/CLOCKS_PER_SEC);
    }
//#pragma acc exit data copyout(error, k)    
    printf("%d %lf\n", k, error);
    free(array);
    free(arraynew);
	

    return 0;
}
