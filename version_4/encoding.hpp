#pragma once

#include <vector>
#include <iostream>
#include <cassert>

using namespace std;

int encodeChar(char c, vector<char> validChar);
char decodeChar(int n, vector<char> validChar);