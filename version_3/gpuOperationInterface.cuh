#pragma once

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define ACCESSROWLEADING2D(row,col,width) ((row * width) + col)
#define ACCESSCOLLEADING2D(row,col,height) ((col * height) + row)

using namespace std;


void cublasGpuDotProduct(double* matrixA, int matrixAHeight, int matrixAWidth, double* matrixB, int matrixBHeight, int matrixBWidth, double* matrixC, bool transposeA, bool transposeB);