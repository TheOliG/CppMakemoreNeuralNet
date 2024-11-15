#pragma once

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cassert>
#include <array>
#include <tuple>
#include "cublas_v2.h"

#define ACCESSROWLEADING2D(row,col,width) ((row * width) + col)
#define ACCESSCOLLEADING2D(row,col,height) ((col * height) + row)

using namespace std;


class CudaMemoryPool{
    public:
        int memoryPoolMaxSize = 100;
        //Contains: allocated memory, the size of that memory and a bool that shows if its being used
        std::array<tuple<double*, size_t, bool>, 100> memoryPoolArray;

        int memoryPoolSize;
        
        //Cuda memory pool, can allocate up to 100 different blocks of memory
        CudaMemoryPool();
        ~CudaMemoryPool();
        double* cudaRequestMemory(size_t requestedSize);
        void unreserveMemory(double* memAddress);      
};


void cublasGpuDotProduct(CudaMemoryPool* memPool, double* matrixA, int matrixAHeight, int matrixAWidth, double* matrixB, int matrixBHeight, int matrixBWidth, double* matrixC, bool transposeA, bool transposeB);


