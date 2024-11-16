#include "CompGraph.hpp"

CompGraph::CompGraph(){
    this->gpuMemPool = new CudaMemoryPool();
}

CompGraph::~CompGraph(){
    delete(gpuMemPool);
}

void CompGraph::forwardPass(){
    //Go backwards through the topological graph
    for(int i = 0; i<forwardPassVec.size(); i++){
        //Invoke the function
        forwardPassVec.at(i)();
    }
}

void CompGraph::backwardPass(){
    //Go backwards through the topological graph
    for(int i = backwardPassVec.size()-1; i>=0; i--){
        //Invoke the function
        backwardPassVec.at(i)();
    }
}

void CompGraph::addToForwardPass(function<void()> func){
    forwardPassVec.push_back(func);
}

void CompGraph::addToBackwardsPass(function<void()> func){
    backwardPassVec.push_back(func);
}

void CompGraph::resetVisitedGradients(){
    for(Node* node : visitedNodes){
        node->resetGrads();
    }
}