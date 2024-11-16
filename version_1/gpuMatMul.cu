#include "gpuMatMul.cuh"
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <cassert>

using namespace std;

//TODO: Optimise memory loading by doing patrial matrix multiplications
__global__
void matrixMul(double* matrixA, double* matrixB, double* outputMatrix, int matrixAHeight, int matrixBWidth, int matrixAWidth)
{   
    //Because we set up the blocks and threads to match with matrix C we can access the current row by looking at the y dimention
    int currentRow = (blockIdx.y * blockDim.y) + threadIdx.y;
    //The same can be done to get the current collumn by accessing the x dimention
    int currentCol = (blockIdx.x * blockDim.x) + threadIdx.x;

    //We need to check if it is in bounds, it is normal for it to be out of bounds when not a square number
    if(currentRow < matrixAHeight && currentCol < matrixBWidth){
        double tempTotal = 0;
        //Loop over the dimentions and calculate the total
        for(int i = 0; i<matrixAWidth; i++){
            tempTotal += matrixA[(currentRow * matrixAWidth) + i] * matrixB[(i * matrixBWidth) + currentCol];
        }
        //Write to the output
        outputMatrix[(currentRow * matrixBWidth) + currentCol] = tempTotal;
    }
    //else{assert(false);}
}

//Randomises the values in a matrix between 0 and 1;
void randomiseMatrix(double *matrix, int matrixHeight, int matrixWidth){
    for(int i = 0; i<matrixHeight*matrixWidth; i++){
        matrix[i] = (double)(rand() % 100)/100.;
    }
}

void testMatrixMul(double *matrixA, double *matrixB, double *expectedOutput, int matrixAHeight, int matrixBWidth, int matrixAWidth){
    double temp;
    for(int i = 0; i<matrixAHeight; i++){
        for(int j = 0; j<matrixBWidth; j++){
            temp = 0;
            for(int k = 0; k<matrixAWidth; k++){
                temp += matrixA[(i * matrixAWidth) + k] * matrixB[(k * matrixBWidth) + j];
            }
            assert(expectedOutput[(i * matrixBWidth) + j] == temp);
        }
    }
}


//Dot product of matrix a with b which results in matrix c
void gpuMatMul(double *matrixA, int matrixAHeight, int matrixAWidth, double *matrixB, int matrixBHeight, int matrixBWidth, double *matrixC, bool matrixATransposed, bool matrixBTransposed){
    
    //The output matrix would look like (matrixAHeight x matrixBWidth), height and width are swapped when transposed though
    int matrixCHeight, matrixCWidth;
    if(matrixATransposed){
        matrixCHeight = matrixAWidth;
    }
    else{
        matrixCHeight = matrixAHeight;
    }
    if(matrixBTransposed){
        matrixCWidth = matrixBHeight;
    }
    else{
        matrixCWidth = matrixBWidth;
    }

    //cout<<"Allocating Space on the gpu"<<endl;
    //Allocate memory
    size_t matrixABytes = matrixAHeight * matrixAWidth * sizeof(double);
    size_t matrixBBytes = matrixBHeight * matrixBWidth * sizeof(double);
    size_t matrixCBytes = matrixCHeight * matrixCWidth * sizeof(double);

    double *cudaMatrixA, *cudaMatrixB, *cudaMatrixC;
    cudaMallocManaged(&cudaMatrixA, matrixABytes);
    cudaMallocManaged(&cudaMatrixB, matrixBBytes);
    cudaMallocManaged(&cudaMatrixC, matrixCBytes);


    //cout<<"Copying memory to gpu"<<endl;
    if(matrixATransposed){
        for(int i = 0; i<matrixAHeight; i++){
            for(int j = 0; j<matrixAWidth; j++){
                //cout<<"Pos: "<<(i * matrixAWidth) + j<<" to pos " << (j * matrixAHeight) + i<<endl;
                cudaMatrixA[(j * matrixAHeight) + i] = matrixA[(i * matrixAWidth) + j];
            }
        }
    }
    else{
        for(int i = 0; i<matrixAHeight * matrixAWidth; i++){
            cudaMatrixA[i] = matrixA[i];
        }
    }
    
    if(matrixBTransposed){
        for(int i = 0; i<matrixBHeight; i++){
            for(int j = 0; j<matrixBWidth; j++){
                //cout<<"Pos: "<<(i * matrixBWidth) + j<<" to pos " << (j * matrixBHeight) + i<<endl;
                cudaMatrixB[(j * matrixBHeight) + i] = matrixB[(i * matrixBWidth) + j];
            }
        }
    }
    else{
        for(int i = 0; i<matrixBHeight * matrixBWidth; i++){
            cudaMatrixB[i] = matrixB[i];
        }
    }
    

    //Define the kernel parameters
    int threadWidth = 16;
    int threadHeight = 16;
    int blockWidth = (matrixCWidth + threadWidth - 1)/threadWidth;
    int blockHeight = (matrixCHeight + threadHeight -1)/threadHeight;
    dim3 threads(threadWidth,threadHeight);
    dim3 blocks(blockWidth, blockHeight);

    //We need to swap the height and width values if any matrices are transformed
    if(matrixATransposed){
        int tempHeight = matrixAHeight;
        matrixAHeight = matrixAWidth;
        matrixAWidth = tempHeight;
    }

    if(matrixBTransposed){
        int tempHeight = matrixBHeight;
        matrixBHeight = matrixBWidth;
        matrixBWidth = tempHeight;
    }


    //Launch the kernel
    //cout<<"Starting gpu matrix multiplication"<<endl;
    matrixMul <<< blocks, threads >>> (cudaMatrixA, cudaMatrixB, cudaMatrixC, matrixCHeight, matrixCWidth, matrixAWidth);

    
    //Wait for the threads to complete
    cudaDeviceSynchronize();
    /*
    cout<<"Testing with cpu matrix multiplication"<<endl;
    testMatrixMul(cudaMatrixA, cudaMatrixB, cudaMatrixC, matrixAHeight, matrixBWidth, matrixAWidth);
    */
    //cout<<"Copying memory back to the main memory"<<endl;
    for(int i = 0; i<matrixCHeight*matrixCWidth; i++){
        matrixC[i] = cudaMatrixC[i];
    }

    //cout<<"Freeing allocated memory"<<endl;
    cudaFree(cudaMatrixA);
    cudaFree(cudaMatrixB);
    cudaFree(cudaMatrixC);

    //cout<<"Success!"<<endl;
}

void testingCudaMatMul(){
    int matrixAHeight = (1<<11);
    int matrixAWidth = 1<<11;//These two values must be equal
    int matrixBHeight = 1<<11;//^
    int matrixBWidth = 1<<11;
    //The output matrix would look like (matrixAHeight x matrixBWidth)

    //cout<<"Allocating Space on the main memory"<<endl;
    //Allocate memory
    size_t matrixABytes = matrixAHeight * matrixAWidth * sizeof(double);
    size_t matrixBBytes = matrixBHeight * matrixBWidth * sizeof(double);
    size_t matrixCBytes = matrixAHeight * matrixBWidth * sizeof(double);
    double *matrixA = (double*)malloc(matrixABytes);
    double *matrixB = (double*)malloc(matrixBBytes);
    double *matrixC = (double*)malloc(matrixCBytes);

    //Initalise the matricies with random values
    //cout<<"Randomising values"<<endl;
    randomiseMatrix(matrixA, matrixAHeight, matrixAWidth);
    randomiseMatrix(matrixB, matrixBHeight, matrixBWidth);


    gpuMatMul(matrixA, matrixAHeight, matrixAWidth, matrixB, matrixBHeight, matrixBWidth, matrixC);
}


