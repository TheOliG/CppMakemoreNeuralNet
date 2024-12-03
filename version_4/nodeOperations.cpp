#include "nodeOperations.hpp"

/*
Embeds the values from the lookupTable at index positions.
Does not calculate gradients for index.
Results in a node of size (indexes->height) x (indexes->width * lookupTable->width) eg:

Indexes: 
[1, 2]
[2, 0]

Lookup table:
[7, 3]
[3, 1]
[0, -2]

Output:
[3, 1, 0, -2]
[0, -2, 7, 3]

*/
void embed(CompGraph* compGraph, Node* indexes, Node* lookupTable, Node* output){
    //Ensure output is the correct size
    output->resize(indexes->height, indexes->width * lookupTable->width);
    
    //Function for the forwards pass
    function<void()> forward = [indexes, lookupTable, output](){
        gpuEncode(
            indexes->cudaValues->cudaMemPtr, indexes->height, indexes->width, 
            lookupTable->cudaValues->cudaMemPtr, lookupTable->height, lookupTable->width,
            output->cudaValues->cudaMemPtr
        );
    };

    //Function for the backwards pass
    function<void()> backward = [indexes, lookupTable, output](){
        gpuEncode(
            indexes->cudaValues->cudaMemPtr, indexes->height, indexes->width, 
            lookupTable->cudaGradients->cudaMemPtr, lookupTable->height, lookupTable->width,
            output->cudaGradients->cudaMemPtr, true
        );
    };
    //Store on the computational graph
    compGraph->addToForwardPass(forward);
    compGraph->addToBackwardsPass(backward);

    //Add Nodes to visited, we dont add Indexes because we dont want to compute gradients for that
    compGraph->visitedNodes.insert(lookupTable);
    compGraph->visitedNodes.insert(output);
}


/*
Averages all the rows and columns, eg:

InputNode:
[1, 2]
[3, 4]

Output:
[2.5]

*/
void average(CompGraph* compGraph, Node* inputNode, Node* outputNode){
    //Resize output node to correct size
    outputNode->resize(1,1);

    function<void()> forward = [inputNode, outputNode](){
        //Resize output node to correct size
        outputNode->resize(1,1);
        gpuAverage(inputNode->cudaValues->cudaMemPtr, inputNode->height, inputNode->width, outputNode->cudaValues->cudaMemPtr);
    };

    function<void()> backward = [inputNode, outputNode](){
        gpuAverage(inputNode->cudaGradients->cudaMemPtr, inputNode->height, inputNode->width, outputNode->cudaGradients->cudaMemPtr, true);
    };
    
    //Add functions to compGraph 
    compGraph->addToForwardPass(forward);
    compGraph->addToBackwardsPass(backward);

    //Add to visited
    compGraph->visitedNodes.insert(inputNode);
    compGraph->visitedNodes.insert(outputNode);
}


/*
Caclulates the dot product between the two nodes,
Outnode cannot be the same as node a and b eg:

MatrixA:
[1, 2]
[3, 4]

Matrix B:
[5]
[6]

Output Matrix: 
[17]
[39]
*/
void dotProduct(CompGraph* cGraph, Node* nodeA, Node* nodeB, Node* outNode){
    //Resize output node
    outNode->resize(nodeA->height, nodeB->width);

    assert(nodeA->width == nodeB->height);
    assert(nodeA != outNode && nodeB != outNode);

    function<void()> forward = [nodeA, nodeB, outNode, cGraph](){
        //Resize output node
        outNode->resize(nodeA->height, nodeB->width);
        cublasGpuDotProduct(nodeA->cudaValues->cudaMemPtr, nodeA->height, nodeA->width, nodeB->cudaValues->cudaMemPtr, nodeB->height, nodeB->width, outNode->cudaValues->cudaMemPtr, false, false);
    };

    function<void()> backward = [nodeA, nodeB, outNode, cGraph](){
        /*
        firstMatrixGradient = resultingMatrixGradient • secondMatrixValues transposed
        secondMatrixGradient = firstMatrixValues transposed • resultingMatrixGradient
        https://youtu.be/dB-u77Y5a6A?si=e_HMJr3RWuZrmuUb&t=3612 
        */
        cublasGpuDotProduct(
            outNode->cudaGradients->cudaMemPtr, outNode->height, outNode->width, 
            nodeB->cudaValues->cudaMemPtr, nodeB->height, nodeB->width, 
            nodeA->cudaGradients->cudaMemPtr, false, true
        );

        //Calculating the second matrix gradients
        cublasGpuDotProduct(
            nodeA->cudaValues->cudaMemPtr, nodeA->height, nodeA->width, 
            outNode->cudaGradients->cudaMemPtr, outNode->height, outNode->width, 
            nodeB->cudaGradients->cudaMemPtr, true, false
        ); 
    };

    cGraph->addToForwardPass(forward);
    cGraph->addToBackwardsPass(backward);

    cGraph->visitedNodes.insert(nodeA);
    cGraph->visitedNodes.insert(nodeB);
    cGraph->visitedNodes.insert(outNode);
}

/*
Will take the given node and its expected outputs, performs softmax 
then it will calculate the negative log of the difference between the expected output and the actual output
Expected values are not added to the computational graph and gradients are not calculated.
inNode and outNode must be seperate 
IMPORTANT: Softmax must not be changed afterwards and gradients for it will not be calculated
eg:

inNode:
[-1.14254, -0.727683]
[0.660491, 1.19685]
expected:
[1, 0]
[0, 1]

OutNode:
[0.921934]
[0.460504]
*/
void crossEntropyLoss(CompGraph* cGraph, Node* inNode, Node* expectedValues, Node* outNode, Node* softmaxNode){
    outNode->resize(inNode->height, 1);
    softmaxNode->resize(inNode->height, inNode->width);

    assert(inNode->height == expectedValues->height && inNode->width == expectedValues->width);
    assert(inNode != outNode);

    /*
    Math for this section is taken from 
    https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    Thank you Thomas Kurbiel :)
    */

    //TODO: Make this run on GPU (currently slow sequential)
    function<void()> forward = [inNode, expectedValues, outNode, softmaxNode](){
        outNode->resize(inNode->height, 1);
        softmaxNode->resize(inNode->height, inNode->width);
        //Loop over all rows
        gpuCrossEntropyLoss(inNode->cudaValues->cudaMemPtr, inNode->height, inNode->width, expectedValues->cudaValues->cudaMemPtr, softmaxNode->cudaValues->cudaMemPtr, outNode->cudaValues->cudaMemPtr);
    };

    function<void()> backward = [inNode, expectedValues, outNode, softmaxNode](){
        gpuCrossEntropyLossBackwards(
            inNode->cudaGradients->cudaMemPtr, inNode->height, inNode->width, \
            expectedValues->cudaValues->cudaMemPtr, 
            softmaxNode->cudaValues->cudaMemPtr,
            outNode->cudaGradients->cudaMemPtr
        );
    };

    cGraph->addToForwardPass(forward);
    cGraph->addToBackwardsPass(backward);

    cGraph->visitedNodes.insert(inNode);
    cGraph->visitedNodes.insert(outNode);
}

/*
Adds a vector to a matrix, vector must have a height of 1 and width the same as inMatrix
eg:

matrix:
[1, 2]
[3, 4]

vector:
[5, 6]

output:
[6, 8]
[8, 10]

*/
void addVector(CompGraph* cGraph, Node* inMatrix, Node* inVector, Node* outNode){
    outNode->resize(inMatrix->height, inMatrix->width);

    assert(inVector->height = 1);
    assert(inVector->width == inMatrix->width);

    function<void()> forward = [inMatrix, inVector, outNode](){
        outNode->resize(inMatrix->height, inMatrix->width);
        gpuAddVector(inMatrix->cudaValues->cudaMemPtr, inMatrix->height, inMatrix->width, inVector->cudaValues->cudaMemPtr, outNode->cudaValues->cudaMemPtr);
    };

    function<void()> backward = [inMatrix, inVector, outNode](){
        gpuAddVector(
            inMatrix->cudaGradients->cudaMemPtr, inMatrix->height, inMatrix->width, 
            inVector->cudaGradients->cudaMemPtr,
            outNode->cudaGradients->cudaMemPtr, true
        );
    };

    cGraph->addToForwardPass(forward);
    cGraph->addToBackwardsPass(backward);

    cGraph->visitedNodes.insert(inMatrix);
    cGraph->visitedNodes.insert(inVector);
    cGraph->visitedNodes.insert(outNode);
}
/*
Performs a tanH operation on the inNode, 
IMPORTANT: outNode values should not be changed after this. 
eg:

inNode:
[1, 2]
[-1, -2]

outNode:
[0.7615942, 0.9640276]
[-0.7615942, -0.9640276]
*/
void tanhOperation(CompGraph* cGraph, Node* inNode, Node* outNode){
    outNode->resize(inNode->height, inNode->width);

    function<void()> forward = [inNode, outNode, cGraph](){
        outNode->resize(inNode->height, inNode->width);
        gpuTanhOperation(inNode->cudaValues->cudaMemPtr, inNode->height, inNode->width, outNode->cudaValues->cudaMemPtr);
    };

    function<void()> backward = [inNode, outNode](){
        gpuTanhOperationBackwards(
            inNode->cudaGradients->cudaMemPtr, inNode->height, inNode->width, 
            outNode->cudaGradients->cudaMemPtr, 
            outNode->cudaValues->cudaMemPtr
        );
    };

    cGraph->addToForwardPass(forward);
    cGraph->addToBackwardsPass(backward);

    cGraph->visitedNodes.insert(inNode);
    cGraph->visitedNodes.insert(outNode);
}