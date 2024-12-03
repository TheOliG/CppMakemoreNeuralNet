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
    checkCublasStatus(cublasDgemm(handle, (cublasOperation_t)transposeA, (cublasOperation_t)transposeB, matrixAHeight, matrixBWidth, matrixAWidth, &alpha, cudaMatrixA, lda, cudaMatrixB, ldb, &beta, cudaMatrixC, ldc));

    //Wait for the threads to complete
    checkCudaStatus(cudaDeviceSynchronize());

    cublasDestroy(handle);
}

__global__
void deviceGpuEncode(double* cudaIndexes, int indexesHeight, int indexesWidth, double* cudaEmbeddingTable, int embeddingHeight, int embeddingWidth, double* cudaOutMatrix, int outHeight, int outWidth){
    //Because we set up the blocks and threads to match with input matrix we can access the current row by looking at the y dimention
    int currentRow = (blockIdx.y * blockDim.y) + threadIdx.y;
    //The same can be done to get the current collumn by accessing the x dimention
    int currentCol = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(currentRow < indexesHeight && currentCol < indexesWidth){
        //Define the current index we are accessing
        int currentIndex = cudaIndexes[ACCESSROWLEADING2D(currentRow, currentCol, indexesWidth)];
        for(int i = 0; i<embeddingWidth; i++){
            cudaOutMatrix[ACCESSROWLEADING2D(currentRow, (currentCol * embeddingWidth) + i, outWidth)] =
                cudaEmbeddingTable[ACCESSROWLEADING2D(currentIndex, i, embeddingWidth)];
        }
    }
}

__global__
void deviceGpuEncodeBackwards(double* cudaIndexes, int indexesHeight, int indexesWidth, double* cudaEmbeddingTableGradients, int embeddingHeight, int embeddingWidth, double* cudaOutMatrixGradients, int outHeight, int outWidth){
    //Because we set up the blocks and threads to match with input matrix we can access the current row by looking at the y dimention
    int currentRow = (blockIdx.y * blockDim.y) + threadIdx.y;
    //The same can be done to get the current collumn by accessing the x dimention
    int currentCol = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(currentRow < indexesHeight && currentCol < indexesWidth){
        //Define the current index we are accessing
        int currentIndex = cudaIndexes[ACCESSROWLEADING2D(currentRow, currentCol, indexesWidth)];
        for(int i = 0; i<embeddingWidth; i++){
            cudaEmbeddingTableGradients[ACCESSROWLEADING2D(currentIndex, i, embeddingWidth)] += 
                cudaOutMatrixGradients[ACCESSROWLEADING2D(currentRow, (currentCol * embeddingWidth) + i, outWidth)];
        }
    }
}


void gpuEncode(double* cudaIndexes, int indexesHeight, int indexesWidth, double* cudaEmbeddingTable, int embeddingHeight, int embeddingWidth, double* cudaOutMatrix, bool backwards){
    
    int outHeight = indexesHeight;
    int outWidth = indexesWidth * embeddingWidth;

    //Define the kernel parameters
    int threadHeight = 16;
    int threadWidth = 16;
    int blockHeight = (indexesHeight + threadHeight -1)/threadHeight;
    int blockWidth = (indexesWidth + threadWidth - 1)/threadWidth;
    dim3 threads(threadWidth,threadHeight);
    dim3 blocks(blockWidth, blockHeight);

    if(!backwards){
        //Launch the kernel
        deviceGpuEncode <<< blocks, threads >>> (cudaIndexes, indexesHeight, indexesWidth, cudaEmbeddingTable, embeddingHeight, embeddingWidth, cudaOutMatrix, outHeight, outWidth);
    }
    else{
        deviceGpuEncodeBackwards <<< blocks, threads >>> (cudaIndexes, indexesHeight, indexesWidth, cudaEmbeddingTable, embeddingHeight, embeddingWidth, cudaOutMatrix, outHeight, outWidth);
    }
    //Wait for the threads to complete
    cudaDeviceSynchronize();
}



__global__
void deviceFullReduce(double* cudaInput, int inputHeight, int inputWidth, double* outPtr){
    //Because we set up the blocks and threads to match with input matrix we can access the current row by looking at the y dimention
    int currentRow = (blockIdx.y * blockDim.y) + threadIdx.y;
    //The same can be done to get the current collumn by accessing the x dimention
    int currentCol = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(currentRow < inputHeight && currentCol < inputWidth){
        (double)atomicAdd(&outPtr[0], cudaInput[ACCESSROWLEADING2D(currentRow, currentCol, inputWidth)]);
    }
}

__global__
void deviceFullReduceBackwards(double* cudaInputGradient, int inputHeight, int inputWidth, double* outPtrGradient){
    //Because we set up the blocks and threads to match with input matrix we can access the current row by looking at the y dimention
    int currentRow = (blockIdx.y * blockDim.y) + threadIdx.y;
    //The same can be done to get the current collumn by accessing the x dimention
    int currentCol = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(currentRow < inputHeight && currentCol < inputWidth){
        cudaInputGradient[ACCESSROWLEADING2D(currentRow, currentCol, inputWidth)] += (outPtrGradient[0]/(double)(inputHeight * inputWidth));
    }
}


void gpuAverage(double* cudaInput, int inputHeight, int inputWidth, double* cudaOutMatrix, bool backwards){
    if(!backwards){
        //Set the output value to zero
        cudaOutMatrix[0] = 0;
    }    
    //Define the kernel parameters
    int threadHeight = 16;
    int threadWidth = 16;
    int blockHeight = (inputHeight + threadHeight -1)/threadHeight;
    int blockWidth = (inputWidth + threadWidth - 1)/threadWidth;
    dim3 threads(threadWidth,threadHeight);
    dim3 blocks(blockWidth, blockHeight);
    if(!backwards){
        //Launch the kernel
        deviceFullReduce <<< blocks, threads >>> (cudaInput, inputHeight, inputWidth, cudaOutMatrix);
    }
    else{
        //Launch the kernel
        deviceFullReduceBackwards <<< blocks, threads >>> (cudaInput, inputHeight, inputWidth, cudaOutMatrix);
    }
    
    //Wait for the threads to complete
    cudaDeviceSynchronize();

    if(!backwards){
        //Calculate average
        cudaOutMatrix[0] = cudaOutMatrix[0] / (double)(inputHeight*inputWidth);
    }
}


__global__
void deviceCrossEntropyLoss(double* cudaInput, int inputHeight, int inputWidth, double* cudaExpected, double* cudaSoftmax, double* cudaOutput){
    //Because we set up the blocks and threads to match with output matrix we can access the current row by looking at the y dimention
    int currentRow = (blockIdx.y * blockDim.y) + threadIdx.y;
    //The same can be done to get the current collumn by accessing the x dimention
    int currentCol = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(currentRow < inputHeight && currentCol < 1){
        //Calculate the max value
        double maxVal = cudaInput[ACCESSROWLEADING2D(currentRow, 0, inputWidth)];
        for(int j = 1; j<inputWidth; j++){
            if(maxVal<cudaInput[ACCESSROWLEADING2D(currentRow, j, inputWidth)]){
                maxVal = cudaInput[ACCESSROWLEADING2D(currentRow, j, inputWidth)];
            }
        }
        
        //Calculate denominator
        double total = 0.;
        for(int j = 0; j<inputWidth; j++){
            total += exp(cudaInput[ACCESSROWLEADING2D(currentRow, j, inputWidth)] - maxVal);
        }

        //Calculate the negative log loss
        double logTotal = 0.;
        for(int j = 0; j<inputWidth; j++){
            cudaSoftmax[ACCESSROWLEADING2D(currentRow, j, inputWidth)] = exp(cudaInput[ACCESSROWLEADING2D(currentRow, j, inputWidth)] - maxVal) / total;
            logTotal += cudaExpected[ACCESSROWLEADING2D(currentRow, j, inputWidth)] * -log(cudaSoftmax[ACCESSROWLEADING2D(currentRow, j, inputWidth)]);
        }
        cudaOutput[currentRow] = logTotal;
    }
}



void gpuCrossEntropyLoss(double* cudaInput, int inputHeight, int inputWidth, double* cudaExpected, double* cudaSoftmax, double* cudaOutput){
    int outputHeight = inputHeight;
    int outputWidth = 1;
    
    //Define the kernel parameters
    int threadHeight = 16;
    int threadWidth = 16;
    int blockHeight = (outputHeight + threadHeight -1)/threadHeight;
    int blockWidth = (outputWidth + threadWidth - 1)/threadWidth;
    dim3 threads(threadWidth,threadHeight);
    dim3 blocks(blockWidth, blockHeight);
    //Launch the kernel
    deviceCrossEntropyLoss <<< blocks, threads >>> (cudaInput, inputHeight, inputWidth, cudaExpected, cudaSoftmax, cudaOutput);
    //Wait for the threads to complete
    cudaDeviceSynchronize();
}

__global__
void deviceCrossEntropyLossBackwards(double* cudaInputGradient, int inputHeight, int inputWidth, double* cudaExpected, double* cudaSoftmax, double* cudaOutputGradient){
    //Because we set up the blocks and threads to match with output matrix we can access the current row by looking at the y dimention
    int currentRow = (blockIdx.y * blockDim.y) + threadIdx.y;
    //The same can be done to get the current collumn by accessing the x dimention
    int currentCol = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(currentRow < inputHeight && currentCol < 1){
        cudaInputGradient[ACCESSROWLEADING2D(currentRow, currentCol, inputWidth)] += 
            (cudaSoftmax[ACCESSROWLEADING2D(currentRow, currentCol, inputWidth)] - cudaExpected[ACCESSROWLEADING2D(currentRow, currentCol, inputWidth)]) 
            * cudaOutputGradient[currentRow];
    }
}

void gpuCrossEntropyLossBackwards(double* cudaInputGradients, int inputHeight, int inputWidth, double* cudaExpected, double* cudaSoftmax, double* cudaOutputGradients){
    
    //Define the kernel parameters
    int threadHeight = 16;
    int threadWidth = 16;
    int blockHeight = (inputHeight + threadHeight -1)/threadHeight;
    int blockWidth = (inputWidth + threadWidth - 1)/threadWidth;
    dim3 threads(threadWidth,threadHeight);
    dim3 blocks(blockWidth, blockHeight);
    //Launch the kernel
    deviceCrossEntropyLossBackwards <<< blocks, threads >>> (cudaInputGradients, inputHeight, inputWidth, cudaExpected, cudaSoftmax, cudaOutputGradients);
    //Wait for the threads to complete
    cudaDeviceSynchronize();
}


__global__
void deviceAddVector(double* cudaInputMatrix, int inputHeight, int inputWidth, double* cudaInputVector , double* cudaOutMatrix){
    //Because we set up the blocks and threads to match with input matrix we can access the current row by looking at the y dimention
    int currentRow = (blockIdx.y * blockDim.y) + threadIdx.y;
    //The same can be done to get the current collumn by accessing the x dimention
    int currentCol = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(currentRow < inputHeight && currentCol < inputWidth){
        cudaOutMatrix[ACCESSROWLEADING2D(currentRow, currentCol, inputWidth)] = cudaInputMatrix[ACCESSROWLEADING2D(currentRow, currentCol, inputWidth)] + cudaInputVector[ACCESSROWLEADING2D(0, currentCol, inputWidth)];
    }
}

__global__
void deviceAddVectorBackwards(double* cudaInputMatrixGradient, int inputHeight, int inputWidth, double* cudaInputVectorGradient, double* cudaOutMatrixGradient){
    //Because we set up the blocks and threads to match with input matrix we can access the current row by looking at the y dimention
    int currentRow = (blockIdx.y * blockDim.y) + threadIdx.y;
    //The same can be done to get the current collumn by accessing the x dimention
    int currentCol = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(currentRow < inputHeight && currentCol < inputWidth){
        cudaInputMatrixGradient[ACCESSROWLEADING2D(currentRow, currentCol, inputWidth)] += cudaOutMatrixGradient[ACCESSROWLEADING2D(currentRow, currentCol, inputWidth)];
        (double)atomicAdd(&cudaInputVectorGradient[currentCol], cudaOutMatrixGradient[ACCESSROWLEADING2D(currentRow, currentCol, inputWidth)]);
    }
}


void gpuAddVector(double* cudaInputMatrix, int inputHeight, int inputWidth, double* cudaInputVector , double* cudaOutMatrix, bool backwards){

    //Define the kernel parameters
    int threadHeight = 16;
    int threadWidth = 16;
    int blockHeight = (inputHeight + threadHeight -1)/threadHeight;
    int blockWidth = (inputWidth + threadWidth - 1)/threadWidth;
    dim3 threads(threadWidth,threadHeight);
    dim3 blocks(blockWidth, blockHeight);
    //Launch the kernel
    if(backwards){
        deviceAddVectorBackwards <<< blocks, threads >>> (cudaInputMatrix, inputHeight, inputWidth, cudaInputVector, cudaInputMatrix);
    }
    else{
        deviceAddVector <<< blocks, threads >>> (cudaInputMatrix, inputHeight, inputWidth, cudaInputVector, cudaOutMatrix);
    }
    //Wait for the threads to complete
    cudaDeviceSynchronize();
}


__global__
void deviceTanhOperation(double* cudaInputMatrix, int inputHeight, int inputWidth, double* cudaOutMatrix){
    //Because we set up the blocks and threads to match with input matrix we can access the current row by looking at the y dimention
    int currentRow = (blockIdx.y * blockDim.y) + threadIdx.y;
    //The same can be done to get the current collumn by accessing the x dimention
    int currentCol = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(currentRow < inputHeight && currentCol < inputWidth){
        cudaOutMatrix[ACCESSROWLEADING2D(currentRow, currentCol, inputWidth)] = tanh(cudaInputMatrix[ACCESSROWLEADING2D(currentRow, currentCol, inputWidth)]);
    }
}




void gpuTanhOperation(double* cudaInputMatrix, int inputHeight, int inputWidth, double* cudaOutMatrix){

    //Define the kernel parameters
    int threadHeight = 16;
    int threadWidth = 16;
    int blockHeight = (inputHeight + threadHeight -1)/threadHeight;
    int blockWidth = (inputWidth + threadWidth - 1)/threadWidth;
    dim3 threads(threadWidth,threadHeight);
    dim3 blocks(blockWidth, blockHeight);
    //Launch the kernel
    deviceTanhOperation <<< blocks, threads >>> (cudaInputMatrix, inputHeight, inputWidth, cudaOutMatrix);
    //Wait for the threads to complete
    cudaDeviceSynchronize();
}

__global__
void deviceTanhOperationBackwards(double* cudaInputMatrixGradient, int inputHeight, int inputWidth, double* cudaOutMatrixGradient, double* cudaOutMatrixValues){
    //Because we set up the blocks and threads to match with input matrix we can access the current row by looking at the y dimention
    int currentRow = (blockIdx.y * blockDim.y) + threadIdx.y;
    //The same can be done to get the current collumn by accessing the x dimention
    int currentCol = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(currentRow < inputHeight && currentCol < inputWidth){
        cudaInputMatrixGradient[ACCESSROWLEADING2D(currentRow, currentCol, inputWidth)] += 
            (1. - pow(cudaOutMatrixValues[ACCESSROWLEADING2D(currentRow, currentCol, inputWidth)], 2)) * 
            cudaOutMatrixGradient[ACCESSROWLEADING2D(currentRow, currentCol, inputWidth)];
    }
}


void gpuTanhOperationBackwards(double* cudaInputMatrixGradients, int inputHeight, int inputWidth, double* cudaOutMatrixGradients, double* cudaOutMatrixValues){

    //Define the kernel parameters
    int threadHeight = 16;
    int threadWidth = 16;
    int blockHeight = (inputHeight + threadHeight -1)/threadHeight;
    int blockWidth = (inputWidth + threadWidth - 1)/threadWidth;
    dim3 threads(threadWidth,threadHeight);
    dim3 blocks(blockWidth, blockHeight);
    //Launch the kernel
    deviceTanhOperationBackwards <<< blocks, threads >>> (cudaInputMatrixGradients, inputWidth, inputHeight, cudaOutMatrixGradients, cudaOutMatrixValues);
    //Wait for the threads to complete
    cudaDeviceSynchronize();
}



__global__
void deviceLearning(double* cudaValues, int height, int width, double* cudaGradients, double learningRate){
    //Because we set up the blocks and threads to match with input matrix we can access the current row by looking at the y dimention
    int row = (blockIdx.y * blockDim.y) + threadIdx.y;
    //The same can be done to get the current collumn by accessing the x dimention
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(row < height && col < width){
        cudaValues[ACCESSROWLEADING2D(row,col,width)] += cudaGradients[ACCESSROWLEADING2D(row,col,width)] * learningRate * -1.;
    }
}


/*
Updates the values by the gradient multiplied by the learning rate and then negated, for reducing the loss
*/
void gpuLearning(double* cudaValues, int height, int width, double* cudaGradients, double learningRate){

    //Define the kernel parameters
    int threadHeight = 16;
    int threadWidth = 16;
    int blockHeight = (height + threadHeight -1)/threadHeight;
    int blockWidth = (width + threadWidth - 1)/threadWidth;
    dim3 threads(threadWidth,threadHeight);
    dim3 blocks(blockWidth, blockHeight);
    //Launch the kernel
    deviceLearning <<< blocks, threads >>> (cudaValues, height, width, cudaGradients, learningRate);
    //Wait for the threads to complete
    cudaDeviceSynchronize();
}