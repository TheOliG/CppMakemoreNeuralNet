#include "CudaMemoryClass.cuh"


CudaMemoryClass::CudaMemoryClass(){
    this->currentAllocatedMemory = 0;
    this->currentUsedMemory = 0;
    this->cudaMemPtr = nullptr;
}

CudaMemoryClass::~CudaMemoryClass(){
    checkCudaStatus(cudaFree(this->cudaMemPtr));
}

double* CudaMemoryClass::setUsedMemory(size_t memSize){
    if(memSize>currentAllocatedMemory){
        
        double* newMemory;
        checkCudaStatus(cudaMallocManaged(&newMemory, memSize));
        if(this->cudaMemPtr != nullptr){
            checkCudaStatus(cudaMemcpy(newMemory ,this->cudaMemPtr, this->currentUsedMemory, cudaMemcpyDeviceToDevice));
            checkCudaStatus(cudaFree(this->cudaMemPtr)); 
        }
        this->cudaMemPtr = newMemory;
        this->currentAllocatedMemory = memSize;
        
    }
    this->currentUsedMemory = memSize;

    return this->cudaMemPtr;
    
}

void CudaMemoryClass::transferToHost(double* dest){
    checkCudaStatus(cudaMemcpy(dest, this->cudaMemPtr, this->currentUsedMemory, cudaMemcpyDeviceToHost));
}

void CudaMemoryClass::transferToDevice(double* src){
    checkCudaStatus(cudaMemcpy(this->cudaMemPtr, src, this->currentUsedMemory, cudaMemcpyHostToDevice));
}