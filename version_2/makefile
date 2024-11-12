main: main.o CompGraph.o Node.o encoding.o gpuOperationInterface.o
	g++ main.o CompGraph.o Node.o encoding.o gpuOperationInterface.o -o main -L/usr/local/cuda/lib64 -lcudart -lcublas -lcuda

main.o: main.cpp
	g++ -c main.cpp -o main.o

CompGraph.o: CompGraph.cpp
	g++ -c CompGraph.cpp -o CompGraph.o

Node.o: Node.cpp
	g++ -c Node.cpp -o Node.o

encoding.o: encoding.cpp
	g++ -c encoding.cpp -o encoding.o

gpuOperationInterface.o: gpuOperationInterface.cu
	nvcc -c gpuOperationInterface.cu -o gpuOperationInterface.o -std=c++14