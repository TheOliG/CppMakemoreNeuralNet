#pragma once

#include <cstdlib>
#include <vector>
#include <cassert>
#include <iostream>

#include "CudaMemoryClass.cuh"

using namespace std;

class Node{
    public:
        //The gradients and value arrays
        double* values;
        double* gradients;

        CudaMemoryClass* cudaValues;
        CudaMemoryClass* cudaGradients;

        //The dimensions of the matrix
        int height;
        int width;

        //This property describes if the node is a trainable parameter
        bool paramNode;

        //Stores the current matrix size, which could be larger than the current height x width x sizeof(double)
        size_t matrixBytes;

        //Constructor
        Node(int height = 1, int width = 1, bool randomiseValues = false, bool parameterNode = false);
        //Deconstructor
        ~Node();

        //Getters
        double& getValue(int row, int col);
        double& getGrad(int row, int col);

        //GPU uploading
        void copyValuesToGpu();
        void copyGradientsToGpu();

        void getValuesFromGpu();
        void getGradientsFromGpu();


        //For resetting values
        void resetValues(double val = 0);
        void resetGrads(double val = 0);

        
        void resize(int height, int width);


        //Prints the matrix
        void print(bool grad = false);
};



