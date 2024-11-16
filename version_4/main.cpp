#include "main.hpp"

int main(){
    //HYPER PARAMETERS 
    int CONTEXT_WINDOW = 4;
    int LOOKUP_DIMENSIONS = 64;
    int HIDDEN_LAYER_SIZE = 2048;
    int NUM_EXAMPLES = 64;
    int NUM_ITER = 2000;
    double LEARNING_RATE = 0.01;


    ifstream inFile;
    inFile.open("../names.txt");
    string line;
    vector<vector<int>> allData;
    vector<char> validChar = {'.','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};
    srand(1);

    while(getline(inFile, line)){
        vector<int> singleWord;
        for(char c : line){
            singleWord.push_back(encodeChar(c, validChar));
        }
        singleWord.push_back(encodeChar('.', validChar));
        allData.push_back(singleWord);
    }


    vector<pair<vector<int>, int>> allDataFormatted;

    for(vector<int> singleWord : allData){
        for(int i = 0; i < singleWord.size(); i++){
            //Initalise the pair and vector
            pair<vector<int>, char> tempPair;
            vector<int> tempVec;
            tempPair.second = singleWord.at(i);

            //Loop over the context window to give context to the tempVec
            for(int j = CONTEXT_WINDOW; j > 0;  j--){
                if((i - j)<0){
                    tempVec.push_back(encodeChar('.', validChar));
                }
                else{
                    tempVec.push_back(singleWord.at(i-j));
                }
            }

            //Push to the whole vector 
            tempPair.first = tempVec;
            allDataFormatted.push_back(tempPair);
        }
    }
    /*
    cout<<"Printing"<<endl;
    for(pair<vector<int>, int> tempPair : allDataFormatted){
        for(int curInt : tempPair.first){
            cout<<decodeChar(curInt, validChar)<<" ";
        }
        cout<<" --> "<<decodeChar(tempPair.second, validChar)<<endl;
    }
    */


    CompGraph* cGraph = new CompGraph();

    
    //Params: 
    Node* lookupTable = new Node(validChar.size(), LOOKUP_DIMENSIONS, true, true);
    Node* hiddenLayerInWeights = new Node(LOOKUP_DIMENSIONS * CONTEXT_WINDOW, HIDDEN_LAYER_SIZE, true, true);
    Node* hiddenLayerOutWeights = new Node(HIDDEN_LAYER_SIZE, validChar.size(), true, false);
    Node* hiddenLayerInBiases = new Node(1, HIDDEN_LAYER_SIZE, false, true);
    Node* logitBiases = new Node(1, validChar.size(), false, true);

    vector<Node*> params = {lookupTable, hiddenLayerInWeights, hiddenLayerOutWeights, hiddenLayerInBiases, logitBiases};

    for(int i = 0; i<hiddenLayerOutWeights->height; i++){
        for(int j = 0; j<hiddenLayerOutWeights->width; j++){
            hiddenLayerOutWeights->getValue(i,j) *= 0.01;
        }
    }

    hiddenLayerInWeights->getValue(0,1) += 20.;

    //Special nodes:
    Node* indexes = new Node(NUM_EXAMPLES, CONTEXT_WINDOW);
    Node* expectedValues = new Node(NUM_EXAMPLES, validChar.size());

    //Nodes for calculation:
    Node* embededNode = new Node();
    Node* hiddenLayerOutput = new Node();
    Node* logitNode = new Node();
    Node* softmaxNode = new Node();
    Node* logLosses = new Node();
    Node* loss = new Node();

    //Construct the computational graph
    embed(cGraph, indexes, lookupTable, embededNode);
    dotProduct(cGraph, embededNode, hiddenLayerInWeights, hiddenLayerOutput);
    addVector(cGraph, hiddenLayerOutput, hiddenLayerInBiases, hiddenLayerOutput);
    tanhOperation(cGraph, hiddenLayerOutput, hiddenLayerOutput);
    dotProduct(cGraph, hiddenLayerOutput, hiddenLayerOutWeights, logitNode);
    addVector(cGraph, logitNode, logitBiases, logitNode);
    crossEntropyLoss(cGraph, logitNode, expectedValues, logLosses, softmaxNode);
    average(cGraph, logLosses, loss);


    double totalForwardTime = 0.;
    double totalBackwardTime = 0.;
    double totalUpdateTime = 0;

    for(int curItter = 0; curItter<=NUM_ITER; curItter++){
        for(int i = 0; i<NUM_EXAMPLES; i++){
            pair<vector<int>, int> curExample = allDataFormatted.at(rand() % allDataFormatted.size());
            for(int j = 0; j<curExample.first.size(); j++){
                indexes->getValue(i, j) = curExample.first.at(j);
            }
            expectedValues->getValue(i, curExample.second) = 1;
        }

        auto t1 = chrono::high_resolution_clock::now();
        cGraph->forwardPass();
        loss->getGrad(0,0) = 1.;
        auto t2 = chrono::high_resolution_clock::now();
        cGraph->backwardPass();
        auto t3 = chrono::high_resolution_clock::now();
        for(Node* param : params){
            for(int i = 0; i<param->height; i++){
                for(int j = 0; j<param->width; j++){
                    if((double)curItter/(double)NUM_ITER < 0.5){
                        param->getValue(i,j) += -LEARNING_RATE * param->getGrad(i,j);
                    }
                    else{
                        param->getValue(i,j) += -LEARNING_RATE * param->getGrad(i,j) * 0.5;
                    }
                }
            }
        }
        auto t4 = chrono::high_resolution_clock::now();

        chrono::duration<double, milli> forwardPassTime = t2 - t1;
        chrono::duration<double, milli> backwardPassTime = t3 - t2;
        chrono::duration<double, milli> updateTime = t4 - t3;

        totalForwardTime+=forwardPassTime.count();
        totalBackwardTime+=backwardPassTime.count();
        totalUpdateTime+=updateTime.count();
        
        if(curItter % 100 == 0){

            cout<<curItter<<"/"<<NUM_ITER<<endl;
            
            cout<<"Average Forward pass time: "<<totalForwardTime/(double)(curItter + 1.)<<endl;
            cout<<"Average Backward pass time: "<<totalBackwardTime/(double)(curItter + 1.)<<endl;
            cout<<"Average Update pass time: "<<totalUpdateTime/(double)(curItter + 1.)<<endl;
            /*
            cout<<"Forward pass time: "<<forwardPassTime.count()<<endl;
            cout<<"Backward pass time: "<<backwardPassTime.count()<<endl;
            cout<<"Update time: "<<updateTime.count()<<endl;
            */
            loss->print();
        }
        
        cGraph->resetVisitedGradients();
        indexes->resetValues();
        expectedValues->resetValues();
    }


    indexes->resize(1, indexes->width);
    expectedValues->resize(1, expectedValues->width);
    
    indexes->resetValues();

    while (true)
    {
        cGraph->forwardPass();

        double randNum = (double)rand()/(double)RAND_MAX;
        double tempTotal = 0.;
        for(int i = 0; i<softmaxNode->width; i++){
            tempTotal += softmaxNode->getValue(0,i);
            if(tempTotal >= randNum){
                if(i != 0){
                    for(int j = 0; j<indexes->width-1; j++){
                        indexes->getValue(0,j) = indexes->getValue(0,j+1);
                    }
                    indexes->getValue(0,indexes->width - 1) = i;
                    cout<<decodeChar(i, validChar);
                }
                else{
                    indexes->resetValues();
                    cout<<endl;
                }
                break;
            }
        }
    }
    
}

