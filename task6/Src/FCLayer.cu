#include <iostream>
#include <cstdio>

#include <cuda_runtime.h>

#include "../Inc/FCLayer.cuh"
#include "../Inc/Errors.cuh"


// CUDA kernel to apply the sigmoid function to each element in the array
__global__
void _sigmoid(float* data, int size)
{
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x < size)
    {
        // Apply the sigmoid function to the current element
        data[x] = 1.0 / (1.0 + expf(-data[x]));
    }
}

// Host function to call the sigmoid kernel
__host__
void sigmoid(float* data, int size)
{
    // Set the number of threads per block
    size_t threads = 16;

    // Calculate the number of blocks needed based on the array size
    size_t blocks = std::ceil(1.0 * size / threads);

    // Set the block size and grid size for the kernel
    dim3 blockSize(threads);
    dim3 gridSize(blocks);

    // Launch the sigmoid kernel
    _sigmoid<<<gridSize, blockSize>>>(data, size);
}

LinearArguments::LinearArguments(const char* pathToWeights, const char* pathToBias, int inSize, int outSize) :
        _pathToWeights(pathToWeights), _pathToBias(pathToBias), _inSize(inSize), _outSize(outSize)  {}

const char* LinearArguments::getPathToWeights() { return this->_pathToWeights; }

const char* LinearArguments::getPathToBias() { return this->_pathToBias; }

int LinearArguments::getInputSize() { return this->_inSize; }

int LinearArguments::getOutputSize() { return this->_outSize; }


Linear::Linear(cublasHandle_t handle, LinearArguments args) : cublasHandle(handle)
{
    this->sizeX = args.getOutputSize();
    this->sizeY = args.getInputSize();

    // Allocate memory
    float* tempBufferForWeights;
    float* tempBufferForBias;
    GET_CUDA_STATUS(cudaMallocHost(&tempBufferForWeights, sizeof(float) * this->sizeY * this->sizeX));
    GET_CUDA_STATUS(cudaMallocHost(&tempBufferForBias, sizeof(float)*this->sizeX));
    GET_CUDA_STATUS(cudaMalloc(&this->weights, sizeof(float) * this->sizeY * this->sizeX));
    GET_CUDA_STATUS(cudaMalloc(&this->bias, sizeof(float)  * this->sizeX));
    GET_CUDA_STATUS(cudaMalloc(&this->output, sizeof(float) * this->sizeX));

    // Here we will write weights from 'pathToWeights' file
    FILE* f_in_weights = std::fopen(args.getPathToWeights(), "rb");
    if (!f_in_weights)
    {
        std::cout << "There's no such file: " << args.getPathToWeights() << std::endl;
        std::exit(-1);
    }

    FILE* f_in_bias = std::fopen(args.getPathToBias(), "rb");
    if (!f_in_bias)
    {
        std::cout << "There's no such file: " << args.getPathToBias() << std::endl;
        std::exit(-1);
    }

    std::fread(tempBufferForWeights, sizeof(float), this->sizeY * this->sizeX, f_in_weights);
    std::fread(tempBufferForBias, sizeof(float), this->sizeX, f_in_bias);

    GET_CUDA_STATUS(cudaMemcpy(
            (void*)this->weights,
            (void*)tempBufferForWeights,
            sizeof(float) * this->sizeY * this->sizeX,
            cudaMemcpyHostToDevice));
    GET_CUDA_STATUS(cudaMemcpy(
            (void*)this->bias,
            (void*)tempBufferForBias,
            sizeof(float) * this->sizeX,
            cudaMemcpyHostToDevice));

    // Delete temp buffer
    GET_CUDA_STATUS(cudaFreeHost(tempBufferForWeights));
    GET_CUDA_STATUS(cudaFreeHost(tempBufferForBias));
    std::fclose(f_in_weights);
    std::fclose(f_in_bias);
}

Linear::~Linear()
{
    if (this->output)   GET_CUDA_STATUS(cudaFree(this->output));
    if (this->weights)  GET_CUDA_STATUS(cudaFree(this->weights));
    if (this->bias)  GET_CUDA_STATUS(cudaFree(this->bias));
}

void Linear::forward(float* input, float** output)
{
    const float alpha = 1.0, beta = 0.0;
    //matrix multiplying
    GET_CUBLAS_STATUS(cublasSgemv_v2(
            this->cublasHandle,
            CUBLAS_OP_T,
            this->sizeY,
            this->sizeX,
            &alpha,
            this->weights,
            this->sizeY,
            input,
            1,
            &beta,
            this->output,
            1));
    //adding bias
    GET_CUBLAS_STATUS(cublasSaxpy(this->cublasHandle, this->sizeX, &alpha, this->bias, 1, this->output, 1));
    *output = this->output;
}

int Linear::getInputSize()
{
    return this->sizeY;
}

int Linear::getOutputSize()
{
    return this->sizeX;
}