#include "Node.hpp"

//Constructor for a scalar node (0 dimentional, just a single value)
//if parameter node is true then it will not be deleted on cleanup
Node::Node(double scalarValue, bool parameterNode){
    this->height = 1;
    this->width = 1;
    this->parameterNode = parameterNode;
    this->matrixValues = (double*)malloc(1 * 1 * sizeof(double));
    this->matrixGradients = (double*)malloc(1 * 1 * sizeof(double));   
    this->matrixValues[0] = scalarValue;
    this->matrixGradients[0] = 0.; 
}

//Constructor for a matrix, (2 dimentional, height and width).
//if parameter node is true then it will not be deleted on cleanup
Node::Node(int height, int width, bool randomiseValues, bool parameterNode){
    //Allocate memory
    this->matrixValues = (double*) malloc(height*width*sizeof(double));
    this->matrixGradients = (double*) malloc(height*width*sizeof(double));
    this->height = height;
    this->width = width;
    this->parameterNode = parameterNode;
    //Populate the matrix
    for(int i = 0; i<height*width; i++){
        double tempVal = 0;
        if(randomiseValues){
            //Random value between -2 and 2
            tempVal = (((double)rand()/(double)RAND_MAX)*2.)-1.;
        }
        this->matrixValues[i] = tempVal;
        //Initialise the value for the gradients
        this->matrixGradients[i] = 0.;
    }
}


//Deconstructor
Node::~Node(){
    delete matrixValues;
    delete matrixGradients;
}


/*
Addition for two nodes of the same size, can also add if otherNode has a height and/or width of 1. eg:
[1, 2]
[3, 4]
\+
[5, 6]
\=
[7, 8]
[8, 10]
*/
Node* Node::add(Node* otherNode, CompGraph* compGraph){

    //Creating the new node 
    Node* newNode = new Node(this->height, this->width);

    //Add the values together
    for(int i = 0; i<this->height; i++){
        for(int j = 0; j<this->width; j++){
            if(otherNode->height == 1 && otherNode->width == 1){
                //Update the value
                newNode->matrixValues[(i * newNode->width) + j] = this->matrixValues[(i * this->width) + j] + otherNode->matrixValues[0];
            }
            else if(otherNode->height == 1){
                //Check the dimentions match
                assert(otherNode->width == this->width);
                //Update value
                newNode->matrixValues[(i * newNode->width) + j] = this->matrixValues[(i * this->width) + j] + otherNode->matrixValues[j];
            }
            else if(otherNode->width == 1){
                //Check the dimentions match
                assert(otherNode->height == this->height);
                //Update the value
                newNode->matrixValues[(i * newNode->width) + j] = this->matrixValues[(i * this->width) + j] + otherNode->matrixValues[i];
            }
            else{
                //Check the dimentions match
                assert(otherNode->height == this->height && otherNode->width == this->width);
                //Update the value
                newNode->matrixValues[(i * newNode->width) + j] = this->matrixValues[(i * this->width) + j] + otherNode->matrixValues[(i * otherNode->width) + j];
            }
        }
    }

    //Addition can be seen as a gradient distributor to the two parents
    function<void()> backwardsFunction = [otherNode, this, newNode](){
        //Add the values together
    for(int i = 0; i<this->height; i++){
        for(int j = 0; j<this->width; j++){
            if(otherNode->height == 1 && otherNode->width == 1){
                //Update the value
                this->matrixGradients[(i * this->width) + j] += newNode->matrixGradients[(i * newNode->width) + j];
                otherNode->matrixGradients[0] += newNode->matrixGradients[(i * newNode->width) + j];
            }
            else if(otherNode->height == 1){
                this->matrixGradients[(i * this->width) + j] += newNode->matrixGradients[(i * newNode->width) + j];
                otherNode->matrixGradients[j] += newNode->matrixGradients[(i * newNode->width) + j];
            }
            else if(otherNode->width == 1){
                this->matrixGradients[(i * this->width) + j] += newNode->matrixGradients[(i * newNode->width) + j];
                otherNode->matrixGradients[i] += newNode->matrixGradients[(i * newNode->width) + j];
            }
            else{
                this->matrixGradients[(i * this->width) + j] += newNode->matrixGradients[(i * newNode->width) + j];
                otherNode->matrixGradients[(i * otherNode->width) + j] += newNode->matrixGradients[(i * newNode->width) + j];
            }
        }
    }
        
    };
    //Add to the computational graph
    compGraph->addToGraph(newNode, backwardsFunction);

    //Add to the visited (for cleanup)
    compGraph->visitedNodes.insert(this);
    compGraph->visitedNodes.insert(otherNode);

    return newNode;
}

//Multiplication, works for: 0d * 0d, 
Node* Node::multiply(Node* otherNode, CompGraph* compGraph){
    //Logic for one scalar multiplied to another scalar (0d * 0d = 0d)
    if(this->height * this->width == 1 && otherNode->width * otherNode->height == 1){
        //Creating the new scalar node 
        Node* newNode = new Node(otherNode->matrixValues[0] * this->matrixValues[0]);
        Node* selfPtr = this;
        //Multiplication can be seen as a gradient "swapper" to the two parents
        function<void()> backwardsFunction = [otherNode, selfPtr, newNode](){
            selfPtr->matrixGradients[0] += otherNode->matrixValues[0] * newNode->matrixGradients[0];
            otherNode->matrixGradients[0] += selfPtr->matrixValues[0] * newNode->matrixGradients[0];
        };
        //Add to the computational graph
        compGraph->addToGraph(newNode, backwardsFunction);

        //Add to the visited (for cleanup)
        compGraph->visitedNodes.insert(this);
        compGraph->visitedNodes.insert(otherNode);

        return newNode;
    }
    cout<<"Error: Cannot multiply larger than 1 x 1 size Node"<<endl;
    assert(false);
}


//Dot product, works for: 2d • 2d. otherNode height must equal this nodes width, transpose does not work for backwards pass!
Node* Node::dotProduct(Node* otherNode, CompGraph* compGraph, bool transposeFirst, bool tranposeSecond){
    //Logic for 2d matrix to 2d matrix dot product 
    int firstTransposedHeight, firstTransposedWidth, secondTransposedHeight, secondTransposedWidth;
    if(transposeFirst){
        firstTransposedHeight = this->width;
        firstTransposedWidth = this->height;
    }
    else{
        firstTransposedHeight = this->height;
        firstTransposedWidth = this->width;
    }
    if(tranposeSecond){
        secondTransposedHeight = otherNode->width;
        secondTransposedWidth = otherNode->height;
    }
    else{
        secondTransposedHeight = otherNode->height;
        secondTransposedWidth = otherNode->width;
    }

    //Checking dimentions
    if(firstTransposedWidth != secondTransposedHeight){
        cout<<"Error: Cannot dot product "<<firstTransposedHeight<<" x "<<firstTransposedWidth<<" matrix with "<<secondTransposedHeight<<" x "<<secondTransposedWidth<<" matrix"<<endl;
        assert(false);
    }
    //Construct the output node
    Node* newNode = new Node(firstTransposedHeight, secondTransposedWidth);


    //Perform matrix multiplication on the GPU
    cublasGpuDotProduct(
        this->matrixValues, this->height, this->width, 
        otherNode->matrixValues, otherNode->height, otherNode->width, 
        newNode->matrixValues, transposeFirst, tranposeSecond
    );


    //We can rename some variables to make it easier to understand
    Node* firstMatrix = this;
    Node* secondMatrix = otherNode;
    Node* resultingMatrix = newNode;
    function<void()> backwardsFunction = [firstMatrix, secondMatrix, resultingMatrix](){
        /*
            firstMatrixGradient = resultingMatrixGradient • secondMatrixValues transposed
            secondMatrixGradient = firstMatrixValues transposed • resultingMatrixGradient
            https://youtu.be/dB-u77Y5a6A?si=e_HMJr3RWuZrmuUb&t=3612 
        */
        //Calculating first matrix gradients
        cublasGpuDotProduct(
            resultingMatrix->matrixGradients, resultingMatrix->height, resultingMatrix->width,
            secondMatrix->matrixValues, secondMatrix->height, secondMatrix->width,
            firstMatrix->matrixGradients, false, true
        );

        //Calculating the second matrix gradients
        cublasGpuDotProduct(
            firstMatrix->matrixValues, firstMatrix->height, firstMatrix->width,
            resultingMatrix->matrixGradients, resultingMatrix->height, resultingMatrix->width,
            secondMatrix->matrixGradients, true, false
        );
    };

    //Making sure to add to the computational graph
    compGraph->addToGraph(newNode, backwardsFunction);
    compGraph->visitedNodes.insert(this);"i should propose to my girlfriend, give her something happy to think of and look forward to, no pressure...also watch my tiktoks";
    compGraph->visitedNodes.insert(otherNode);

    return newNode;
}

//Performs a tanh function on the computational graph
Node* Node::tanh(CompGraph* compGraph){
    //Output node
    Node* outNode = new Node(this->height, this->width);

    //Update all the values
    for(int i = 0; i< this->height * this->width; i++){
        outNode->matrixValues[i] = std::tanh(this->matrixValues[i]);
    }


    //Gradient of tanh(x) is 1-tanh^2(x)
    function<void()> backwardsFunc = [this, outNode](){
        for(int i = 0; i< (this->width * this->height); i++){
            this->matrixGradients[i] = (1. - pow(outNode->matrixValues[i], 2)) * outNode->matrixGradients[i];
        }
    };


    compGraph->addToGraph(outNode, backwardsFunc);
    compGraph->visitedNodes.insert(this);
    
    return outNode;
}


Node* Node::batchNorm(CompGraph* compGraph){
    return nullptr;
}


Node* Node::getNode(int row, int col, CompGraph* compGraph){

    //Check in bounds
    assert(row<this->height && col<this->width);

    //Initialise the new node with the value
    Node* outNode = new Node(this->matrixValues[(row * this->width) + col]);

    //Define the backwards function (it simply copies the gradient)
    Node* thisPtr = this;
    function<void()> tempFunc = [thisPtr, outNode, row, col](){
        thisPtr->matrixGradients[(row * thisPtr->width) + col] += outNode->matrixGradients[0];
    };

    //Add to computaional graph
    compGraph->addToGraph(outNode, tempFunc);
    compGraph->visitedNodes.insert(this);

    return outNode;
}

//Gets a column of a 2d node as a new 2d node
Node* Node::getCollumnNode(int col, CompGraph* compGraph){
    //Check in bounds
    assert(col<this->width);

    //Create a new node
    Node* outNode = new Node(this->height, 1);

    //Copy the values
    for(int i = 0; i<this->height; i++){
        outNode->matrixValues[i] = this->matrixValues[(i * this->width)+col];
    }

    //Define the backwards function (it simply copies the gradient)
    Node* thisPtr = this;
    function<void()> tempFunc = [thisPtr, outNode, col](){
        for(int i = 0; i<thisPtr->height; i++){
            thisPtr->matrixGradients[(i * thisPtr->width)+col] += outNode->matrixGradients[i];
        }
    };

    //Add to the computational graph
    compGraph->addToGraph(outNode, tempFunc);
    compGraph->visitedNodes.insert(this);

    return outNode;
}

//Gets a the row of a 2d node as a new 2d node
Node* Node::getRowNode(int row, CompGraph* compGraph){
    //Check in bounds
    assert(row<this->height);

    //Create a new node
    Node* outNode = new Node(1, this->width);

    //Copy the values
    for(int i = 0; i<this->width; i++){
        outNode->matrixValues[i] = this->matrixValues[(row * this->width) + i];
    }

    //Define the backwards function (it simply copies the gradient)
    Node* thisPtr = this;
    function<void()> tempFunc = [thisPtr, outNode, row](){
        for(int i = 0; i<thisPtr->width; i++){
            thisPtr->matrixGradients[(row * thisPtr->width)+i] += outNode->matrixGradients[i];
        }
    };

    //Add to the computational graph
    compGraph->addToGraph(outNode, tempFunc);
    compGraph->visitedNodes.insert(this);

    return outNode;
}

/*
Will take the given node and its expected outputs, perform softmax and 
then it will calculate the negative log of the difference between the expected output and the actual output
eg:

[1, 2]
[3, 4]
expected:
[0, 1]
[1, 0]
\=
[2.531]
[4.1234]
(Example values)
*/

Node* Node::crossEntropyLoss(Node* expectedOutputs ,CompGraph* compGraph){

    assert(this->width == expectedOutputs->width && this->height == expectedOutputs->height);
    Node* outNode = new Node(this->height, 1);
    Node* softmaxNode = new Node(this->height, this->width);

    //Itterate over all the rows
    for(int curRow = 0; curRow < this->height; curRow++){
        //Store the total and max value found
        double maxValue = this->matrixValues[(curRow * this->width)];
        double tempTotal = 0.;

        //Itterate over all columns
        for(int curCol = 0; curCol < this->width; curCol++){
            //Update the max value
            if(this->matrixValues[(curRow * this->width) + curCol] > maxValue){
                maxValue = this->matrixValues[(curRow * this->width) + curCol];
            }
        }

        //Itterate over all columns
        for(int curCol = 0; curCol < this->width; curCol++){
            //Update total for denominator (we reduce the value by the max in order to keep the values small and usable)
            tempTotal += exp(this->matrixValues[(curRow * this->width) + curCol] - maxValue);
        }

        //Itterate over all columns
        for(int curCol = 0; curCol < this->width; curCol++){
            softmaxNode->matrixValues[(curRow * softmaxNode->width) + curCol] = exp(this->matrixValues[(curRow * this->width) + curCol] - maxValue)/tempTotal;

            //Formula: https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/ 
            outNode->matrixValues[(curRow * outNode->width) + 0] += log(softmaxNode->matrixValues[(curRow * softmaxNode->width) + curCol]) * expectedOutputs->matrixValues[(curRow * expectedOutputs->width) + curCol] * -1;
        }

        
    }

    Node* thisPtr = this;
    //Math: https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1 
    function<void()> tempFunc = [thisPtr, outNode, softmaxNode, expectedOutputs](){
        for(int i = 0; i<thisPtr->height; i++){
            for(int j = 0; j<thisPtr->width; j++){
                thisPtr->matrixGradients[(i * thisPtr->width) + j] += (softmaxNode->matrixValues[(i * softmaxNode->width) + j] - expectedOutputs->matrixValues[(i * expectedOutputs->width) + j]) * outNode->matrixGradients[(i * outNode->width) + 0];
            }
        }
    };

    //Add to the graph
    compGraph->addToGraph(outNode, tempFunc);
    compGraph->visitedNodes.insert(this);
    compGraph->visitedNodes.insert(softmaxNode);
    compGraph->visitedNodes.insert(expectedOutputs);

    return outNode;
}


/*
Gets the mean of each column and transforms the height to 1, eg:

[1, 2]
[3, 4]
\=
[2, 3]
*/
Node* Node::averageColumns(CompGraph* compGraph){
    Node* outNode = new Node(1, this->width);
    
    //Itterate over all the columns
    for(int i = 0; i<this->width; i++){
        //Itterate over all the rows
        for(int j = 0; j<this->height; j++){
            //Calculate the sum
            outNode->matrixValues[(0 * outNode->width) + i] += this->matrixValues[(j * this->width) + i];
        }
        //Cacluate the average
        outNode->matrixValues[(0 * outNode->width) + i] *= (1./this->height);
    }

    Node* thisPtr = this;

    function<void()> backwardsFunc = [thisPtr, outNode](){
        for(int i = 0; i<thisPtr->height; i++){
            for(int j = 0; j<thisPtr->width; j++){
                //Gradient is 1/height
                thisPtr->matrixGradients[(i * thisPtr->width) + j] += (1./thisPtr->height) * outNode->matrixGradients[(0 * outNode->width) + j];
            }
        }
    };

    //Add to computational graph
    compGraph->addToGraph(outNode, backwardsFunc);
    compGraph->visitedNodes.insert(this);

    return outNode;
}


/*
Gets the mean of each row and transforms the width to 1, eg:

[1, 2]
[3, 4]
\=
[1.5]
[3.5]
*/
Node* Node::averageRows(CompGraph* compGraph){
    Node* outNode = new Node(this->height, 1);
    
    //Itterate over all the rows
    for(int i = 0; i<this->height; i++){
        //Itterate over all the columns
        for(int j = 0; j<this->width; j++){
            //Calculate the sum
            outNode->matrixValues[(i * outNode->width) + 0] += this->matrixValues[(i * this->width) + j];
        }
        //Cacluate the average
        outNode->matrixValues[(i * outNode->width) + 0] *= (1./this->width);
    }

    Node* thisPtr = this;

    function<void()> backwardsFunc = [thisPtr, outNode](){
        for(int i = 0; i<thisPtr->height; i++){
            for(int j = 0; j<thisPtr->width; j++){
                //Gradient is 1/height
                thisPtr->matrixGradients[(i * thisPtr->width) + j] += (1./thisPtr->width) * outNode->matrixGradients[(i * outNode->width) + 0];
            }
        }
    };

    //Add to computational graph
    compGraph->addToGraph(outNode, backwardsFunc);
    compGraph->visitedNodes.insert(this);

    return outNode;
}


/*
Concat the two Nodes horizontally, eg:
[1, 2]               
[3. 4]
concat
[5, 6]
[7, 8]
\=
[1, 2, 5, 6]
[3, 4, 7, 8]
The two Nodes must have the same height
*/

Node* Node::concatHorizontally(Node* otherNode, CompGraph* compGraph){
    //Check that the heights match
    assert(otherNode->height == this->height);

    Node* outNode = new Node(this->height, this->width + otherNode->width);
    //Itterate all the rows
    for(int i = 0; i<this->height; i++){
        //Adding this node
        for(int j = 0; j<this->width; j++){
            outNode->matrixValues[(i * outNode->width) + j] = this->matrixValues[(i * this->width) + j];
        }

        //Adding other node
        for(int j = 0; j<otherNode->width; j++){
            outNode->matrixValues[(i * outNode->width) + (j + this->width)] = otherNode->matrixValues[(i * otherNode->width) + (j)];
        }
    }

    Node* thisPtr = this;
    function<void()> backFunc = [thisPtr, otherNode, outNode](){
        //Itterate over all the rows
        for(int i = 0; i<thisPtr->height; i++){
            //Go over all the columns of "this"
            for(int j = 0; j<thisPtr->width; j++){
                thisPtr->matrixGradients[(i * thisPtr->width) + j] += 1. * outNode->matrixGradients[(i * outNode->width) + j];
            }

            //Go over all the columns of "other"
            for(int j = 0; j<otherNode->width; j++){
                otherNode->matrixGradients[(i * thisPtr->width) + j] += 1. * outNode->matrixGradients[(i * outNode->width) + (j + thisPtr->width)];
            }
        }
    };

    //Add to the computational graph
    compGraph->addToGraph(outNode, backFunc);
    compGraph->visitedNodes.insert(this);
    compGraph->visitedNodes.insert(otherNode); 

    return outNode;
}



/*
Concat the two Nodes verticaly, eg:
[1, 2]
[3. 4] 
concat
[5, 6]
[7, 8]
\=
[1, 2]
[3. 4] 
[5, 6]
[7, 8]

The two Nodes must have the same width
*/
Node* Node::concatVerticaly(Node* otherNode, CompGraph* compGraph){
    //Check that the widths match
    assert(otherNode->width == this->width);
    Node* outNode = new Node(this->height+otherNode->height, this->width);
    //Itterate all the columns
    for(int curCol = 0; curCol<this->width; curCol++){
        //Adding this node
        for(int curRow = 0; curRow<this->height; curRow++){
            outNode->matrixValues[(curRow * outNode->width) + curCol] = this->matrixValues[(curRow * this->width) + curCol];
        }

        //Adding other node
        for(int curRow = 0; curRow<otherNode->height; curRow++){
            outNode->matrixValues[((curRow + this->height) * outNode->width) + curCol] = otherNode->matrixValues[(curRow * otherNode->width) + curCol];
        }
    }

    Node* thisPtr = this;
    function<void()> backFunc = [thisPtr, otherNode, outNode](){
        //Itterate all the columns
        for(int curCol = 0; curCol<thisPtr->width; curCol++){
            //Adding this node
            for(int curRow = 0; curRow<thisPtr->height; curRow++){
                thisPtr->matrixGradients[(curRow * thisPtr->width) + curCol] += outNode->matrixGradients[(curRow * outNode->width) + curCol];
            }

            //Adding other node
            for(int curRow = 0; curRow<otherNode->height; curRow++){
                otherNode->matrixGradients[(curRow * otherNode->width) + curCol] += outNode->matrixGradients[((curRow + thisPtr->height) * outNode->width) + curCol];
            }
        }
    };

    //Add to the computational graph
    compGraph->addToGraph(outNode, backFunc);
    compGraph->visitedNodes.insert(this);
    compGraph->visitedNodes.insert(otherNode); 

    return outNode;
}


//Runs backwards calculation on the whole graph
void Node::backwards(CompGraph* compGraph){
    //Set the gradients to 1. (base case)
    for(int i = 0; i<this->height * this->width; i++){
        this->matrixGradients[i] = 1.;
    }
    //Go backwards through the entire graph
    compGraph->backwards();
}

//Prints the matrix values for debugging
void Node::print(){
    //Show the dimentions
    cout<<"Node " << this->height <<" x "<<this->width<<":"<<endl;
    for(int i = 0; i<this->height; i++){
        cout<<"[ ";
        for(int j = 0; j<this->width; j++){
            cout<<this->matrixValues[(i*this->width)+j];
            if(j != this->width-1){
                cout<<", ";
            }
        }
        cout<<" ]"<<endl;
    }
}


Node* concatNodes(vector<Node*> nodeVector, CompGraph* compGraph, bool concatHorizontaly){
    //Check that the size is greater than 0
    assert(nodeVector.size()>0);

    //Set the current node to the first in the vector
    Node* curNode = nodeVector.at(0);
    
    bool first = true;
    //Loop over all the nodes
    for(Node* node : nodeVector){
        if(first){
            first = false;
        }
        else{
            if(concatHorizontaly){
                curNode = curNode->concatHorizontally(node, compGraph);
            }
            else{
                curNode = curNode->concatVerticaly(node, compGraph);
            }
        }
    }

    return curNode;
}


