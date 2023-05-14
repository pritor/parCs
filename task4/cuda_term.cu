#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
////////////////////////////////////////////////////////////////////////////
//расчет уравнения теплопроводности по блокам и потокам. Используем j-ый поток и i-ый блок при каждом расчете, принцип с четвертьюсумммой остается тем же
////////////////////////////////////////////////////////////////////////////
__global__ void heat_equation(double* u, double* u_new, int size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if(!((j == 0 || i == 0 || j == size - 1 || i == size - 1))
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

    if(idx>0)
        return;

    out[idx] = fabs(u_new[idx] - u[idx]);
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
    size_t realsize = size*size;

////////////////////////////////////////////////////////////////////////////
//расчет граничных условий, начало отсчета тактов
////////////////////////////////////////////////////////////////////////////
    clock_t start = clock();

    for (int i = 1; i < size-1; i++) {
        array[i] = array[0] + step * i;
        array[size * (size - 1) + i] = array[(size - 1) * size] + step*i;
        array[size * i] = array[0] + step*i;
        array[(size - 1) + i * size] = array[size - 1] + step * i;
    }
////////////////////////////////////////////////////////////////////////////
//копируем из рассчитанной матрицы в новую
////////////////////////////////////////////////////////////////////////////
    std::memcpy(arr, arr_new, realsize * sizeof(double));
    cudaSetDevice(1);
////////////////////////////////////////////////////////////////////////////
//объявление указателей под матрицы на устройстве, матрицы для ошибок  и буфер, а также последующее выделение памяти для них на устройстве
////////////////////////////////////////////////////////////////////////////

    double* Matrix, *MatrixNew, *error_matrix, *device_error, *error_temp = 0;

    cudaError_t cudaStatus_1 = cudaMalloc((void**)(&Matrix), sizeof(double) * realsize);
    cudaError_t cudaStatus_2 = cudaMalloc((void**)(&MatrixNew), sizeof(double) * realsize);
    cudaMalloc((void**)&device_error, sizeof(double));
    cudaStatus_1 = cudaMalloc((void**)&error_matrix, sizeof(double) * realsize);

    if (cudaStatus_1 != 0 || cudaStatus_2 != 0)
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
    cudaMalloc((&error_temp), temp_size);
////////////////////////////////////////////////////////////////////////////
//создание графа
////////////////////////////////////////////////////////////////////////////
    int k = 0;

    double* error;
    cudaMallocHost(&error, sizeof(double);
    *error = 1.0;

    bool isGraphCreated = false;
    cudaStream_t stream, memoryStream;
    cudaStreamCreate(&stream);
    cudaStreamCreate(&memoryStream);
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

            cudaMemcpyAsync(&error_matrix, device_error, sizeof(double), cudaMemcpyDeviceToHost, memoryStream);

            cudaStreamSynchronize(stream);

            k += 100;
        }
            ////////////////////////////////////////////////////////////////////////////
            //вызов расчета функции теплопроводности на устройстве с размером сетки size-1 и таким же размером блока
            ////////////////////////////////////////////////////////////////////////////
        else
        {
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            for(size_t i = 0; i < 100; i++)
            {
                heat_equation<<<gridDim, blockDim, 0, stream>>>(Matrix, MatrixNew, size);
                k++;

            }
            ////////////////////////////////////////////////////////////////////////////
            //вызов нахождения ошибки на устройстве с тем же размером сетки и блока
            ////////////////////////////////////////////////////////////////////////////
            get_error<<<gridSize, blockSize, 0, stream>>>(Matrix, MatrixNew, Error);
            cub::DeviceReduce::Max(error_temp, temp_size, error_matrix, device_error, realsize);

            double* temp = Matrix;
            Matrix = MatrixNew;
            MatrixNew = temp;

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
    cudaStreamDestroy(memoryStream);
    cudaGraphDestroy(graph);

    return 0;

}