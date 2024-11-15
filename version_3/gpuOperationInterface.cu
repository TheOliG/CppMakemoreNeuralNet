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
void cublasGpuDotProduct(double* matrixA, int matrixAHeight, int matrixAWidth, double* matrixB, int matrixBHeight, int matrixBWidth, double* matrixC, bool transposeA, bool transposeB){
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
    checkCudaStatus(cudaMallocManaged(&cudaMatrixA, matrixABytes));
    checkCudaStatus(cudaMallocManaged(&cudaMatrixB, matrixBBytes));
    checkCudaStatus(cudaMallocManaged(&cudaMatrixC, matrixCBytes));


    //Transfering the matricies to the device (gpu), this also changes the format to column leading
    for(int i = 0; i<matrixAHeight; i++){
        for(int j = 0; j<matrixAWidth; j++){
            cudaMatrixA[ACCESSCOLLEADING2D(i, j, matrixAHeight)] = matrixA[ACCESSROWLEADING2D(i, j, matrixAWidth)];
            //cout<<matrixA[ACCESSROWLEADING2D(i, j, matrixAWidth)]<<", ";
        }
        //cout<<endl;
    }

    for(int i = 0; i<matrixBHeight; i++){
        for(int j = 0; j<matrixBWidth; j++){
            cudaMatrixB[ACCESSCOLLEADING2D(i, j, matrixBHeight)] = matrixB[ACCESSROWLEADING2D(i, j, matrixBWidth)];
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
            matrixC[ACCESSROWLEADING2D(i, j, matrixCWidth)] = cudaMatrixC[ACCESSCOLLEADING2D(i, j, matrixCHeight)];
        }
    }

    //Free memory
    cudaFree(cudaMatrixA);
    cudaFree(cudaMatrixB);
    cudaFree(cudaMatrixC);
    cublasDestroy(handle);
}



//nvcc -o gpuOperationInterface gpuOperationInterface.cu -lcublas -lcuda


/*
//Main for testing
int main (void){

    int matrixAHeight = 3;
    int matrixAWidth = 2;
    int matrixBHeight = 3; 
    int matrixBWidth = 3;
    int matrixCHeight = 2;
    int matrixCWidth = 3;

    size_t matrixABytes = matrixAHeight * matrixAWidth * sizeof(double);
    size_t matrixBBytes = matrixBHeight * matrixBWidth * sizeof(double);
    size_t matrixCBytes = matrixCHeight * matrixCWidth * sizeof(double);

    double* matrixA = (double*)malloc(matrixABytes);
    double* matrixB = (double*)malloc(matrixBBytes);
    double* matrixC = (double*)malloc(matrixCBytes);


    
    for(int i = 0; i<matrixAHeight*matrixAWidth; i++){
        matrixA[i] = i;
    }
    for(int i = 0; i<matrixBHeight*matrixBWidth; i++){
        matrixB[i] = i;
    }


    cublasGpuDotProduct(matrixA, matrixAHeight, matrixAWidth, matrixB, matrixBHeight, matrixBWidth, matrixC, true, true);

    for(int i = 0; i<matrixCHeight; i++){
        for(int j = 0; j<matrixCWidth; j++){
            cout<<matrixC[ACCESSROWLEADING2D(i, j, matrixCWidth)]<<", ";
        }
        cout<<endl;
    }

    free(matrixA);
    free(matrixB);
    free(matrixC);   
}
*/