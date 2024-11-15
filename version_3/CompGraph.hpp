#pragma once
#include <vector>
#include <functional>
#include <unordered_set>

#include "Node.hpp"

using namespace std;


class CompGraph {
    public:
        //A set of all the Node ptrs used for cleanup and gradient reset
        unordered_set<Node*> visitedNodes;
        
        //The vector containing the functions used to compute the forward pass in topological order
        vector<function<void()>> forwardPassVec;

        //The vector containing the fuctions used to compute gradients, topological order
        vector<function<void()>> backwardPassVec;




        void forwardPass();
        void backwardPass();
        void addToForwardPass(function<void()> func);
        void addToBackwardsPass(function<void()> func);
        void resetVisitedGradients();
};