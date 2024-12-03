#pragma once

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <array>
#include <tuple>
#include "cublas_v2.h"

#define ACCESSROWLEADING2D(row,col,width) ((row * width) + col)
#define ACCESSCOLLEADING2D(row,col,height) ((col * height) + row)

using namespace std;

void checkCudaStatus(cudaError_t status);
void checkCublasStatus(cublasStatus_t status);

void cublasGpuDotProduct(double* cudaMatrixA, int matrixAHeight, int matrixAWidth, double* cudaMatrixB, int matrixBHeight, int matrixBWidth, double* cudaMatrixC, bool transposeA, bool transposeB);
void gpuEncode(double* cudaIndexes, int indexesHeight, int indexesWidth, double* cudaEmbeddingTable, int embeddingHeight, int embeddingWidth, double* cudaOutMatrix, bool backwards = false);
void gpuAverage(double* cudaInput, int inputHeight, int inputWidth, double* cudaOutMatrix, bool backwards = false);
void gpuCrossEntropyLoss(double* cudaInput, int inputHeight, int inputWidth, double* cudaExpected, double* cudaSoftmax, double* cudaOutput);
void gpuCrossEntropyLossBackwards(double* cudaInputGradients, int inputHeight, int inputWidth, double* cudaExpected, double* cudaSoftmax, double* cudaOutputGradients);
void gpuAddVector(double* cudaInputMatrix, int inputHeight, int inputWidth, double* cudaInputVector , double* cudaOutMatrix,bool backwards = false);
void gpuTanhOperation(double* cudaInputMatrix, int inputHeight, int inputWidth, double* cudaOutMatrix);
void gpuTanhOperationBackwards(double* cudaInputMatrixGradients, int inputHeight, int inputWidth, double* cudaOutMatrixGradients, double* cudaOutMatrixValues);
void gpuLearning(double* cudaValues, int height, int width, double* cudaGradients, double learningRate);
