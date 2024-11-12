#pragma once
#include <cstdlib>
#include <iostream>
#include <functional>
#include <vector>
#include <cassert>
#include <cmath>
#include "CompGraph.hpp"
#include "gpuOperationInterface.cuh"

using namespace std;
class Node{
    public:
        //For 2 dimentional matrix Nodes
        //Should be of size (height x width)
        double* matrixValues;
        //Should be of size (height x width)        
        double* matrixGradients;
        int height;
        int width;
        bool parameterNode;


        //Constructors
        Node(double scalarValue, bool parameterNode = false);
        Node(int height, int width, bool randomiseValues = false, bool parameterNode = false);
        //Deconstructor
        ~Node();

        void backwards(CompGraph* CompGraph);



        Node* add(Node* otherNode, CompGraph* compGraph);
        Node* multiply(Node* otherNode, CompGraph* compGraph);
        Node* dotProduct(Node* otherNode, CompGraph* compGraph, bool transposeFirst = false, bool tranposeSecond = false);
        Node* tanh(CompGraph* CompGraph);
        Node* batchNorm(CompGraph* compGraph);
        Node* crossEntropyLoss(Node* expectedOutputs ,CompGraph* compGraph);
        Node* averageColumns(CompGraph* compGraph);
        Node* averageRows(CompGraph* compGraph);


        Node* getNode(int row, int col, CompGraph* compGraph);
        Node* getCollumnNode(int col, CompGraph* compGraph);
        Node* getRowNode(int row, CompGraph* compGraph);
        Node* concatHorizontally(Node* otherNode, CompGraph* compGraph);
        Node* concatVerticaly(Node* otherNode, CompGraph* compGraph);
        

        //Debugging
        void print();
};

//Concats the nodes in the vector to eachother verticaly, will concat horizontaly if it is set to true
Node* concatNodes(vector<Node*> nodeVector, CompGraph* CompGraph, bool concatHorizontaly = false);