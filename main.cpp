#include "main.hpp"
#include "Node.hpp"
#include "CompGraph.hpp"
#include "encoding.hpp"

vector<double> softMax(double* inputVector, int width);


int main(void){
    //HYPER PARAMETERS 
    int CONTEXT_WINDOW = 4;
    int LOOKUP_DIMENSIONS = 32;
    int HIDDEN_LAYER_SIZE = 256;
    int NUM_EXAMPLES = 64;
    int NUM_ITER = 20000;
    double LEARNING_RATE = 0.01;


    ifstream inFile;
    inFile.open("names.txt");
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

    CompGraph* graph = new CompGraph();


    //Define our parameters
    Node* lookupTable = new Node(validChar.size(), LOOKUP_DIMENSIONS, true, true);

    Node* hiddenLayerInWeights = new Node(CONTEXT_WINDOW*LOOKUP_DIMENSIONS, HIDDEN_LAYER_SIZE, true, true);
    Node* hiddenLayerOutWeights = new Node(HIDDEN_LAYER_SIZE, validChar.size(), true, true);

    Node* hiddenLayerInBiases = new Node(1, HIDDEN_LAYER_SIZE, false, true);
    Node* outLayerBiases = new Node(1, validChar.size(), false, true);

    //Make output weights small so that the values are close to zero
    for(int i = 0; i<hiddenLayerOutWeights->width*hiddenLayerOutWeights->height; i++){
        hiddenLayerOutWeights->matrixValues[i] *= 0.01;
    }

    vector<Node*> params = {lookupTable, hiddenLayerInWeights, hiddenLayerOutWeights, outLayerBiases, hiddenLayerInBiases};

    vector<pair<vector<int>, int>> trainingSet = allDataFormatted;

    for(int currentIter = 0; currentIter<=NUM_ITER; currentIter++){
        Node* expectedOutput = new Node(NUM_EXAMPLES, validChar.size());
        vector<Node*> inLayerVector;
        for(int i = 0; i<NUM_EXAMPLES; i++){
            //Randomly select an example from the training set
            pair<vector<int>, int> curTrainingExample = trainingSet.at(rand() % trainingSet.size());
            //Setting the expected output with one hot encoding
            expectedOutput->matrixValues[(i * expectedOutput->width) + curTrainingExample.second] = 1.;
            vector<Node*> tempSingleExample;
            for(int j = 0; j<CONTEXT_WINDOW; j++){
                tempSingleExample.push_back(lookupTable->getRowNode(curTrainingExample.first.at(j), graph));
            }
            //Concat the vector Nodes horizontaly
            inLayerVector.push_back(concatNodes(tempSingleExample, graph, true));
        }
        
        //Concat the vector nodes verticaly
        Node* inLayerNode = concatNodes(inLayerVector, graph, false);
        Node* hiddenLayerNode = inLayerNode->dotProduct(hiddenLayerInWeights, graph)->tanh(graph)->add(hiddenLayerInBiases, graph);
        Node* logits = hiddenLayerNode->dotProduct(hiddenLayerOutWeights, graph)->add(outLayerBiases, graph);
        Node* crossEntropyLoss = logits->crossEntropyLoss(expectedOutput, graph);
        Node* loss = crossEntropyLoss->averageColumns(graph);
        
        loss->backwards(graph);
        //Update the parameters given the gradients they have
        for(Node* param : params){
            for(int i = 0; i<param->width * param->height; i++){
                if(currentIter/NUM_ITER > 0.5){
                    param->matrixValues[i] += -LEARNING_RATE * param->matrixGradients[i] * 0.1;
                }
                else{
                    param->matrixValues[i] += -LEARNING_RATE * param->matrixGradients[i];
                }
            }
        }

        if(currentIter % 10 == 0){
            cout<<currentIter<<"/"<<NUM_ITER<<endl;
            loss->print();
        }
        //Free memory and cleanup the comp graph
        graph->cleanup();
    }

    
    vector<int> zeroedContext;
    for(int i = 0; i<CONTEXT_WINDOW; i++){
        zeroedContext.push_back(0);
    }
    vector<int> currentContext = zeroedContext;

    for(int curIter = 1; curIter<INT32_MAX; curIter++){

        vector<Node*> tempSingleExample;
        for(int j = 0; j<CONTEXT_WINDOW; j++){
            tempSingleExample.push_back(lookupTable->getRowNode(currentContext.at(j), graph));
        }
        //Concat the vector Nodes horizontaly
        Node* inLayerNode = concatNodes(tempSingleExample, graph, true);
        Node* hiddenLayerNode = inLayerNode->dotProduct(hiddenLayerInWeights, graph)->add(hiddenLayerInBiases, graph);
        Node* logits = hiddenLayerNode->dotProduct(hiddenLayerOutWeights, graph)->add(outLayerBiases, graph);
        vector<double> output = softMax(logits->matrixValues, logits->width);


        double randNum = (double)rand()/(double)RAND_MAX;
        double tempTotal = 0;
        int choice;
        for(int i = 0; i<output.size(); i++){
            tempTotal += output.at(i); 
            if(tempTotal>=randNum){
                choice = i;
                break;
            }
        }
        if(choice == 0){
            cout<<endl;
            currentContext = zeroedContext;
        }
        else{
            cout<<decodeChar(choice, validChar);
            for(int i = 1; i<CONTEXT_WINDOW; i++){
                currentContext.at(i-1) = currentContext.at(i);
            }
            currentContext.at(currentContext.size()-1) = choice;
        }

        if(curIter%1000 == 0){
            string temp;
            cin>>temp;
        }

    }
}




vector<double> softMax(double* inputVector, int width){

    double max = -INTMAX_MAX;
    for(int i = 0; i<width; i++){
        if(max < inputVector[i]){
            max = inputVector[i];
        }
    }

    double total = 0.;
    for(int i = 0; i<width; i++){
        total += exp(inputVector[i] - max);
    }

    vector<double> out;
    for(int i = 0; i<width; i++){
        out.push_back(exp(inputVector[i] - max)/total);
    }

    return out;
}


