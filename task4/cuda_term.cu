#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

////////////////////////////////////////////////////////////////////////////
//расчет уравнения теплопроводности по блокам и потокам. Используем j-ый поток и i-ый блок при каждом расчете, принцип с четвертьюсумммой остается тем же
////////////////////////////////////////////////////////////////////////////
__global__ void heat_equation(double* u, double* u_new, int size) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i * size + j > size * size) return;
    if(!((j == 0 || i == 0 || j == size - 1 || i == size - 1)))
    {
        u_new[i * size + j] = 0.25 * (u[(i - 1) * size + j] + u[(i + 1) * size + j]+
                                  u[i * size + (j - 1)] + u[i * size + (j + 1)]);
    }
}
////////////////////////////////////////////////////////////////////////////
//расчет ошибки по потокам, проставление его в итоговую матрицу
////////////////////////////////////////////////////////////////////////////
__global__ void get_error(double* u, double* u_new, double* out, int size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx>size*size)
        return;

    out[idx] = std::abs(u_new[idx] - u[idx]);
}

void printGrid(double* array, int size){
    for(int i=0;i<size;i++)printf("-");
    printf("\n");
    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
            printf("%lf ", array[j+i*size]);

        }
        printf("\n");
        printf("\n");
    }
    printf("\n");
    for(int i=0;i<size;i++) printf("-");
    printf("\n");
}

////////////////////////////////////////////////////////////////////////////
//основное тело программы, принимает два аргумента - количество этих аргументов, и массив массивов чаров - аргументы
////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
////////////////////////////////////////////////////////////////////////////
//приведение аргументов к нужному типу
////////////////////////////////////////////////////////////////////////////
    double accuracy;
    int size, iternum;
    accuracy = atof(argv[1]);
    size = atoi(argv[2]);
    iternum = atoi(argv[3]);

    std::cout << "Parameters: " << std::endl <<
              "Min error: " << accuracy << std::endl <<
              "Grid size: " << size << std::endl<<
              "Maximal number of iteration: " << iternum << std::endl ;

    size_t realsize = size*size;

////////////////////////////////////////////////////////////////////////////
//выделение памяти
////////////////////////////////////////////////////////////////////////////
    double* arr;
    double* arr_new;

    cudaMallocHost(&arr, realsize * sizeof(double));
    cudaMallocHost(&arr_new, realsize * sizeof(double));

    std::memset(arr, 0, realsize * sizeof(double));

////////////////////////////////////////////////////////////////////////////
//проставление краевых условий
////////////////////////////////////////////////////////////////////////////
    arr[0] = 10.0;
    arr[size-1] = 20.0;
    arr[(size-1) *size] = 20.0;
    arr[size * size-1] = 30.0;


    double step = 10.0/(size-1);


////////////////////////////////////////////////////////////////////////////
//расчет граничных условий, начало отсчета тактов
////////////////////////////////////////////////////////////////////////////


    for (int i = 1; i < size-1; i++) {
        arr[i] = arr[0] + step * i;
        arr[size * (size - 1) + i] = arr[(size - 1) * size] + step*i;
        arr[size * i] = arr[0] + step*i;
        arr[(size - 1) + i * size] = arr[size - 1] + step * i;
    }
    printGrid(arr, size);

////////////////////////////////////////////////////////////////////////////
//копируем из рассчитанной матрицы в новую
////////////////////////////////////////////////////////////////////////////
    std::memcpy(arr_new, arr, realsize * sizeof(double));
//    cudaSetDevice(1);
////////////////////////////////////////////////////////////////////////////
//объявление указателей под матрицы на устройстве, матрицы для ошибок  и буфер, а также последующее выделение памяти для них на устройстве
////////////////////////////////////////////////////////////////////////////

    double* Matrix, *MatrixNew, *error_matrix, *device_error, *error_temp = 0;

    cudaError_t cudaStatus_1 = cudaMalloc((void**)(&Matrix), sizeof(double) * realsize);
    cudaError_t cudaStatus_2 = cudaMalloc((void**)(&MatrixNew), sizeof(double) * realsize);
    cudaMalloc((void**)&device_error, sizeof(double));
    cudaError_t cudaStatus_3 = cudaMalloc((void**)&error_matrix, sizeof(double) * realsize);

    if (cudaStatus_1 != 0 || cudaStatus_2 != 0 || cudaStatus_3 != 0)
    {
        std::cout << "Memory allocation error" << std::endl;
        return -1;
    }

////////////////////////////////////////////////////////////////////////////
//копируем из рассчитанных матриц хоста на устройство
////////////////////////////////////////////////////////////////////////////
    cudaStatus_1 = cudaMemcpy(Matrix, arr, realsize*sizeof(double), cudaMemcpyHostToDevice);
    cudaStatus_2 = cudaMemcpy(MatrixNew, arr_new, realsize*sizeof(double), cudaMemcpyHostToDevice);

    if (cudaStatus_1 != 0 || cudaStatus_2 != 0)
    {
        std::cout << "Memory transfering error" << std::endl;
        return -1;
    }

    size_t temp_size = 0;

////////////////////////////////////////////////////////////////////////////
//функция редукции определяет, какой размер буфера ей понадобится и записывает его в tempsize
//следующей же строкой - выделение памяти
////////////////////////////////////////////////////////////////////////////
    cub::DeviceReduce::Max(error_temp, temp_size, error_matrix, device_error, realsize);
    cudaMalloc((void**)&error_temp, temp_size);
////////////////////////////////////////////////////////////////////////////
//создание графа
////////////////////////////////////////////////////////////////////////////
    int k = 0;

    double* error;
    cudaMallocHost(&error, sizeof(double));
    *error = 1.0;

    bool isGraphCreated = false;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t graph;
    cudaGraphExec_t instance;
////////////////////////////////////////////////////////////////////////////
//максимальный размер блока - 1024, расчет размера блока в зависимости от GPU, в blockSize будет лежать оптимальный размер сетки
////////////////////////////////////////////////////////////////////////////
    size_t threads = (size < 1024) ? size : 1024;
    unsigned int blocks = size / threads;

    dim3 blockDim(threads / 32, threads / 32);
    dim3 gridDim(blocks * 32, blocks * 32);

    clock_t start = clock();
    for (; (k < iternum) && (*error > accuracy);) {
        if (isGraphCreated)
        {
            cudaGraphLaunch(instance, stream);

            cudaMemcpyAsync(error, device_error, sizeof(double), cudaMemcpyDeviceToHost, stream);

            cudaStreamSynchronize(stream);

            k += 100;
        }
            ////////////////////////////////////////////////////////////////////////////
            //вызов расчета функции теплопроводности на устройстве с размером сетки size-1 и таким же размером блока
            ////////////////////////////////////////////////////////////////////////////
        else
        {
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            for(size_t i = 0; i < 50; i++)
            {
                heat_equation<<<gridDim, blockDim, 0, stream>>>(Matrix, MatrixNew, size);
                heat_equation<<<gridDim, blockDim, 0, stream>>>(MatrixNew, Matrix, size);
            }
            ////////////////////////////////////////////////////////////////////////////
            //вызов нахождения ошибки на устройстве с тем же размером сетки и блока
            ////////////////////////////////////////////////////////////////////////////
            get_error<<<threads * blocks * blocks, threads, 0, stream>>>(Matrix, MatrixNew, error_matrix, size);
            cub::DeviceReduce::Max(error_temp, temp_size, error_matrix, device_error, realsize, stream);

//           std::swap(Matrix, MatrixNew);

            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            isGraphCreated = true;
        }
    }
////////////////////////////////////////////////////////////////////////////
//конец отсчета, вывод полученных данных
////////////////////////////////////////////////////////////////////////////
    clock_t end = clock();
    printf("Time is = %lf\n", 1.0*(end-start)/CLOCKS_PER_SEC);

    printf("%d %lf\n", k, *error);
////////////////////////////////////////////////////////////////////////////
//очистка памяти
////////////////////////////////////////////////////////////////////////////
    cudaFree(Matrix);
    cudaFree(MatrixNew);
    cudaFree(error_matrix);
    cudaFree(error_temp);
    cudaFree(arr);
    cudaFree(arr_new);

    cudaStreamDestroy(stream);
    cudaGraphDestroy(graph);

    return 0;

}