main: main.o Node.o CompGraph.o nodeOperations.o encoding.o gpuOperationInterface.o
	g++ main.o Node.o CompGraph.o nodeOperations.o encoding.o gpuOperationInterface.o -o compiledProgram.o -L/usr/local/cuda/lib64 -lcudart -lcublas -lcuda

main.o: main.cpp
	g++ -c main.cpp -o main.o

Node.o: Node.cpp
	g++ -c Node.cpp -o Node.o

CompGraph.o: CompGraph.cpp
	g++ -c CompGraph.cpp -o CompGraph.o

nodeOperations.o: nodeOperations.cpp
	g++ -c nodeOperations.cpp -o nodeOperations.o

encoding.o: encoding.cpp
	g++ -c encoding.cpp -o encoding.o

gpuOperationInterface.o: gpuOperationInterface.cu
	nvcc -c gpuOperationInterface.cu -o gpuOperationInterface.o -std=c++14