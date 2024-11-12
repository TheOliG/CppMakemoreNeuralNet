main: main.o CompGraph.o Node.o encoding.o gpuMatMul.o
	g++ main.o CompGraph.o Node.o encoding.o gpuMatMul.o -o main -L/usr/local/cuda/lib64 -lcudart

main.o: main.cpp
	g++ -c main.cpp -o main.o

CompGraph.o: CompGraph.cpp
	g++ -c CompGraph.cpp -o CompGraph.o

Node.o: Node.cpp
	g++ -c Node.cpp -o Node.o

encoding.o: encoding.cpp
	g++ -c encoding.cpp -o encoding.o

gpuMatMul.o: gpuMatMul.cu
	nvcc -c gpuMatMul.cu -o gpuMatMul.o -std=c++14