#pragma once

#include <string>

#include <cublas_v2.h>

__host__
void sigmoid(float* data, int size);

struct LinearArguments
{
public:
    LinearArguments(const char* pathToWeights, const char* pathToBias, int inSize, int outSize);

    const char* getPathToWeights();
    const char* getPathToBias();
    int getInputSize();
    int getOutputSize();

private:
    const char* _pathToWeights, *_pathToBias;
    int _inSize, _outSize;
};

class Linear
{
public:
    Linear(cublasHandle_t handle, LinearArguments args);
    ~Linear();

    void forward(float* input, float** output);

    int getInputSize();
    int getOutputSize();

private:
    float* input;
    float* output;
    float* weights;
    float* bias;
    int sizeX, sizeY;
    cublasHandle_t cublasHandle;
};