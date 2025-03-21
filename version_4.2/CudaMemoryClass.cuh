#pragma once

#include <cassert>
#include <iostream>
#include <cuda_runtime.h>

#include "gpuOperationInterface.cuh"

using namespace std;


class CudaMemoryClass{
    public: 
        double* cudaMemPtr;
        size_t currentUsedMemory;//The memory currently being used
        size_t currentAllocatedMemory;//The actual amount of memory allocated by maloc

        CudaMemoryClass();
        ~CudaMemoryClass();

        double* setUsedMemory(size_t memSize);

        void transferToHost(double* dest); //Transfers the data to host, assumes memory is already allocated
        void transferToDevice(double* src); //Transfers the data from src onto the device
};