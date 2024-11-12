#include "encoding.hpp"

int encodeChar(char c, vector<char> validChar){
    for(int i = 0; i<validChar.size(); i++){
        if(validChar.at(i) == c){
            return i;
        }
    }
    cout<<"Error, char: " << c << " could not be encoded"<<endl;
    assert(false);
}

char decodeChar(int n, vector<char> validChar){
    //Check in bounds
    assert(n<validChar.size() && n>=0);

    return validChar.at(n);
}