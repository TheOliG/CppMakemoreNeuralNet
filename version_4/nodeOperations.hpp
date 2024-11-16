#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <array>

#include "CompGraph.hpp"
#include "Node.hpp"
#include "gpuOperationInterface.cuh"


void embed(CompGraph* compGraph, Node* indexes, Node* lookupTable, Node* output);
void average(CompGraph* compGraph, Node* inputNode, Node* outputNode);
void dotProduct(CompGraph* cGraph, Node* nodeA, Node* nodeB, Node* outNode);
void crossEntropyLoss(CompGraph* cGraph, Node* inNode, Node* expectedValues, Node* outNode, Node* softmaxNode);
void addVector(CompGraph* cGraph, Node* inMatrix, Node* vector, Node* outNode);
void tanhOperation(CompGraph* cGraph, Node* inNode, Node* outNode);