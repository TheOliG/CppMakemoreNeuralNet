#include "CompGraph.hpp"
#include "Node.hpp"

CompGraph::CompGraph(){}

//Simply adds the given node to the topological graph
void CompGraph::addToGraph(Node* node, function<void()> backwardsFunction){
    //Add the node to the topological graph
    topologicalGraph.emplace_back(node, backwardsFunction);
    this->visitedNodes.insert(node);
}

void CompGraph::backwards(){
    //Itterate through the topological graph backwards
    for(int i = this->topologicalGraph.size()-1; i>=0; i--){
        //Evoke the backwards pass function in each, the reason we can just go through all is because the gradients should be 0 for the ones that arent the selected node to backwards pass from 
        topologicalGraph.at(i).second();
    }
}

void CompGraph::resetVisitedGradients(){
    //Itterate through all visited nodes
    for(Node* tempNodePtr : this->visitedNodes){
        //Itterate through gradient matrix
        for(int i = 0; i<tempNodePtr->height * tempNodePtr->width; i++){
            tempNodePtr->matrixGradients[i] = 0.;
        }
    }
}


//Deletes the Nodes and frees their respective memory, unless they are a parameter node
void CompGraph::cleanup(){
    //Reset the gradients first
    this->resetVisitedGradients();

    //Itterate through all visited nodes
    for(Node* tempNodePtr : this->visitedNodes){
        if(!(tempNodePtr->parameterNode)){
            delete tempNodePtr;
        }
    }
    this->visitedNodes.clear();
    this->topologicalGraph.clear();
}