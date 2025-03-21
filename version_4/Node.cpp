#include "Node.hpp"



//Constructor
Node::Node(int height, int width, bool randomiseValues, bool parameterNode){
    this->height = height;
    this->width = width;
    this->paramNode = parameterNode;


    //Allocate memory for the arrays
    this->matrixBytes = height*width*sizeof(double);
    this->values = (double*) malloc(this->matrixBytes);
    this->gradients = (double*) malloc(this->matrixBytes);

    //Allocate GPU memory
    this->cudaValues = new CudaMemoryClass();
    this->cudaGradients = new CudaMemoryClass();
    this->cudaValues->setUsedMemory(matrixBytes);
    this->cudaGradients->setUsedMemory(matrixBytes);
    

    
    for(int i = 0; i<height * width; i++){
        double tempVal = 0;
        if(randomiseValues){
            //Random value between -2 and 2
            tempVal = (((double)rand()/(double)RAND_MAX)*2.)-1.;
        }
        this->values[i] = tempVal;
        //Initialise the value for the gradients
        this->gradients[i] = 0.;
    }
}

//Deconstructor
Node::~Node(){
    free(this->values);
    free(this->gradients);
}

void Node::copyValuesToGpu(){
    cudaValues->transferToDevice(values);
}

void Node::copyGradientsToGpu(){
    cudaGradients->transferToDevice(gradients);
}

void Node::getValuesFromGpu(){
    cudaValues->transferToHost(values);
}

void Node::getGradientsFromGpu(){
    cudaGradients->transferToHost(gradients);
}



//Returns a refrence to the value at the given row and col
double& Node::getValue(int row, int col){
    return this->values[ACCESSCOLLEADING2D(row, col, this->height)];
}

//Returns a refrence to the gradient at the given row and col
double& Node::getGrad(int row, int col){
    return this->gradients[ACCESSCOLLEADING2D(row, col, this->height)];
}

//Sets all the values to a given scalar
void Node::resetValues(double val){
    for(int i = 0; i<this->width * this->height; i++){
        this->values[i] = val;
    }
}

//Sets all the gradients to a given scalar
void Node::resetGrads(double val){
    for(int i = 0; i<this->width * this->height; i++){
        this->gradients[i] = val;
    }
}

//This resizes the Node, possibly destroying the data inside it
void Node::resize(int height, int width){
    size_t newSize = height * width * sizeof(double);
    if(newSize > this->matrixBytes){
        //Free the previously allocated memory
        free(this->values);
        free(this->gradients);

        //Realocating the memory
        this->values = (double*) malloc(newSize);
        this->gradients = (double*) malloc(newSize);

        //Setting the new matrixBytes
        this->matrixBytes = newSize;
    }

    //Allocating new space in the gpu
    this->cudaValues->setUsedMemory(newSize);
    this->cudaGradients->setUsedMemory(newSize);

    this->height = height;
    this->width = width;
}

//Prints the matrix values for debugging
void Node::print(bool grad){
    if(grad){
        cout<<"Node " << this->height <<" x "<<this->width<<" Gradients"<<":"<<endl;
    }
    else{
        cout<<"Node " << this->height <<" x "<<this->width<<" Values"<<":"<<endl;
    }
    for(int i = 0; i<this->height; i++){
        cout<<"[ ";
        for(int j = 0; j<this->width; j++){
            if(grad){
                cout<<this->gradients[(i*this->width)+j];
            }
            else{
                cout<<this->values[(i*this->width)+j];
            }
            if(j != this->width-1){
                cout<<", ";
            }
        }
        cout<<" ]"<<endl;
    }
}