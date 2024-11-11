#pragma once
#include <vector>
#include <functional>
#include <unordered_set>

using namespace std;

class Node;

class CompGraph {
    public:
        //This is the computational graph, which includes each node 
        //pair is in format of Node, Backwards function (Parent pointers are stored in the backwards function)
        vector<pair<Node*, function<void()>>> topologicalGraph;

        //A set of all the Node ptrs used for cleanup and gradient reset
        unordered_set<Node*> visitedNodes;

        CompGraph();
        void backwards();
        void resetVisitedGradients();
        void addToGraph(Node* node, function<void()> backwardsFunction);
        void cleanup();
        
};