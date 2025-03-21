#include "gpuOperationInterface.cuh"

//For error checking
void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS API failed!" << std::endl;
        exit(EXIT_FAILURE);
    }
}

//For error checking
void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA API failed! " << cudaGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}



/*
Uses cublas to calculate the dotproduct between two matricies
*/

void cublasGpuDotProduct(double* cudaMatrixA, int matrixAHeight, int matrixAWidth, double* cudaMatrixB, int matrixBHeight, int matrixBWidth, double* cudaMatrixC, bool transposeA, bool transposeB){
    //TODO: Store this handle somewhere so we dont have to keep deleting it (probs in comp graph)
    cublasHandle_t handle;
    cublasCreate(&handle);

    int matrixCHeight = matrixAHeight;
    if(transposeA){
        matrixCHeight = matrixAWidth;
    }

    int matrixCWidth = matrixBWidth;
    if(transposeB){
        matrixCWidth = matrixBHeight;
    }



    // Storing values needed to perform dot product
    double alpha = 1.0;
    double beta = 0.0;

    int lda = matrixAHeight;
    int ldb = matrixBHeight;
    int ldc = matrixCHeight;

    if(transposeA){
        int tempHeight = matrixAHeight;
        matrixAHeight = matrixAWidth;
        matrixAWidth = tempHeight;
    }

    if(transposeB){
        int tempHeight = matrixBHeight;
        matrixBHeight = matrixBWidth;
        matrixBWidth = tempHeight;
    }

    //This is the cublas matrix multiplication algorithm
    checkCublasStatus(cublasDgemm_v2(handle, (cublasOperation_t)transposeA, (cublasOperation_t)transposeB, matrixAHeight, matrixBWidth, matrixAWidth, &alpha, cudaMatrixA, lda, cudaMatrixB, ldb, &beta, cudaMatrixC, ldc));

    //Wait for the threads to complete
    checkCudaStatus(cudaDeviceSynchronize());

    cublasDestroy(handle);
}




void cublasGpuDotProductOld(CudaMemoryPool* memPool, double* matrixA, int matrixAHeight, int matrixAWidth, double* matrixB, int matrixBHeight, int matrixBWidth, double* matrixC, bool transposeA, bool transposeB){
    cublasHandle_t handle;

    cublasCreate(&handle);

    int matrixCHeight = matrixAHeight;
    if(transposeA){
        matrixCHeight = matrixAWidth;
    }

    int matrixCWidth = matrixBWidth;
    if(transposeB){
        matrixCWidth = matrixBHeight;
    }


    size_t matrixABytes = matrixAWidth * matrixAHeight * sizeof(double);
    size_t matrixBBytes = matrixBWidth * matrixBHeight * sizeof(double);
    size_t matrixCBytes = matrixCWidth * matrixCHeight * sizeof(double);

    double *cudaMatrixA, *cudaMatrixB, *cudaMatrixC;
    /*
    checkCudaStatus(cudaMallocManaged(&cudaMatrixA, matrixABytes));
    checkCudaStatus(cudaMallocManaged(&cudaMatrixB, matrixBBytes));
    checkCudaStatus(cudaMallocManaged(&cudaMatrixC, matrixCBytes));
    */

    cudaMatrixA = memPool->cudaRequestMemory(matrixABytes);
    cudaMatrixB = memPool->cudaRequestMemory(matrixBBytes);
    cudaMatrixC = memPool->cudaRequestMemory(matrixCBytes);

    //Transfering the matricies to the device (gpu), this also changes the format to column leading
    for(int i = 0; i<matrixAHeight; i++){
        for(int j = 0; j<matrixAWidth; j++){
            cudaMatrixA[ACCESSCOLLEADING2D(i, j, matrixAHeight)] = matrixA[ACCESSCOLLEADING2D(i, j, matrixAHeight)];
            //cout<<matrixA[ACCESSROWLEADING2D(i, j, matrixAWidth)]<<", ";
        }
        //cout<<endl;
    }

    for(int i = 0; i<matrixBHeight; i++){
        for(int j = 0; j<matrixBWidth; j++){
            cudaMatrixB[ACCESSCOLLEADING2D(i, j, matrixBHeight)] = matrixB[ACCESSCOLLEADING2D(i, j, matrixBHeight)];
            //cout<<matrixB[ACCESSROWLEADING2D(i, j, matrixBWidth)]<<", ";
        }
        //cout<<endl;
    }


    // Storing values needed to perform dot product
    double alpha = 1.0;
    double beta = 0.0;

    int lda = matrixAHeight;
    int ldb = matrixBHeight;
    int ldc = matrixCHeight;

    if(transposeA){
        int tempHeight = matrixAHeight;
        matrixAHeight = matrixAWidth;
        matrixAWidth = tempHeight;
    }

    if(transposeB){
        int tempHeight = matrixBHeight;
        matrixBHeight = matrixBWidth;
        matrixBWidth = tempHeight;
    }

    //This is the cublas matrix multiplication algorithm
    checkCublasStatus(cublasDgemm(handle, (cublasOperation_t)transposeA, (cublasOperation_t)transposeB, matrixAHeight, matrixBWidth, matrixAWidth, &alpha, cudaMatrixA, lda, cudaMatrixB, ldb, &beta, cudaMatrixC, ldc));

    //Wait for the threads to complete
    checkCudaStatus(cudaDeviceSynchronize());

    //Copy the data back into the matrix C array
    for(int i = 0; i<matrixCHeight; i++){
        for(int j = 0; j<matrixCWidth; j++){
            matrixC[ACCESSCOLLEADING2D(i, j, matrixCHeight)] = cudaMatrixC[ACCESSCOLLEADING2D(i, j, matrixCHeight)];
        }
    }

    //Free memory
    /*
    cudaFree(cudaMatrixA);
    cudaFree(cudaMatrixB);
    cudaFree(cudaMatrixC);
    */
    memPool->unreserveMemory(cudaMatrixA);
    memPool->unreserveMemory(cudaMatrixB);
    memPool->unreserveMemory(cudaMatrixC);
    cublasDestroy(handle);
}







CudaMemoryPool::CudaMemoryPool(){
    this->memoryPoolSize = 0;
}

CudaMemoryPool::~CudaMemoryPool(){
    for(int i = 0; i<memoryPoolSize; i++){
        //Check that the memory is not currently being used
        assert(get<2>(this->memoryPoolArray.at(i)) == false);
        //Unallocate all memory
        checkCudaStatus(cudaFree(get<0>(this->memoryPoolArray.at(i))));
    }
}

double* CudaMemoryPool::cudaRequestMemory(size_t requestedSize){
    //Loop through all potential memory blocks
    for(int i = 0; i<memoryPoolSize; i++){
        //Check if its being used
        if(get<2>(this->memoryPoolArray.at(i)) == false){
            get<2>(this->memoryPoolArray.at(i)) = true;
            //Resize if its too small
            if(get<1>(this->memoryPoolArray.at(i)) < requestedSize){
                checkCudaStatus(cudaFree(get<0>(this->memoryPoolArray.at(i))));
                checkCudaStatus(cudaMallocManaged(&get<0>(this->memoryPoolArray.at(i)), requestedSize));
            }
            return get<0>(this->memoryPoolArray.at(i));
        }
    }

    //Create a new memory block
    memoryPoolSize++;
    assert(memoryPoolSize<=100);

    checkCudaStatus(cudaMallocManaged(&get<0>(this->memoryPoolArray.at(memoryPoolSize-1)), requestedSize));
    get<1>(memoryPoolArray.at(memoryPoolSize-1)) = requestedSize;
    get<2>(memoryPoolArray.at(memoryPoolSize-1)) = true;
    return get<0>(this->memoryPoolArray.at(memoryPoolSize-1));
}

void CudaMemoryPool::unreserveMemory(double* memoryAdr){
    //Loop through all potential memory blocks
    for(int i = 0; i<memoryPoolSize; i++){
        if(get<0>(memoryPoolArray.at(i)) == memoryAdr){
            get<2>(memoryPoolArray.at(i)) = false;
            return;
        }
    }
    cout<<"Memory does not match any in pool!"<<endl;
    assert(false);
}
