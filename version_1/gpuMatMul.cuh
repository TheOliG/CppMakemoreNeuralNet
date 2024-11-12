#pragma once



void gpuMatMul(double *matrixA, int matrixAHeight, int matrixAWidth, double *matrixB, int matrixBHeight, int matrixBWidth, double *matrixC, bool matrixATransposed = false, bool matrixBTransposed = false);