# Basic Neural Network in C++
By Oliver Guzowski


This project is a **simple neural network** in **C++** that implements a **character-level language model** inspired by [Andrej Karpathy's "makemore"](https://github.com/karpathy/makemore). The model is built from the ground up, including custom implementations of **backpropagation** using **computational graphs**. This project is implemented with minimal dependencies, using only the **C++ Standard Library** and **CUDA** for GPU acceleration.


## Overview


This project uses a basic neural network approach taken from [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), which utilises embedding to predict the next character in a sequence
- **Forward Pass and Backpropagation**: Efficiently propagates through the network and calculates the gradients through a backward pass.
- **Computational Graphs**: Each computation is represented as a node in a graph (CompGraph.cpp), allowing the model to compute gradients during backpropagation and to manage memory.
- **GPU Acceleration with CUDA**: Critical components are accelerated using CUDA to leverage GPU performance for matrix operations, this greatly improves the performance of the model.


This project is only intended for educational purposes and was done to help understand how NLP neural networks work under the hood.


## Features


- **Automatic gradient calculation**: Class based implementation allows for automated gradient calculation for ease of use and expandability.
- **CUDA Integration**: Matrix operations are optimised with CUDA for faster computation on compatible GPUs.
- **Lightweight Design**: The neural network uses only the C++ standard library and CUDA, with no dependencies on external libraries like TensorFlow or PyTorch.


## Project Structure

- `version_1`
- `version_2`
- `version_3`
   - `main.cpp`: The main file that is used to format the sample data and perform forwards pass.
   - `Node.cpp`: The implementation of a "node" which is used to store gradients and data.
   - `CompGraph.cpp`: The implementation of the computational graph.
   - `encoding.cpp`: Does the character encoding.
   - `gpuOperationInterface.cu`: Custom CuBLAS interface
   - `nodeOperations.cu`: Forward and backward passes for the computational graph
   - `makefile`: The makefile used to compile the project.
- `names.txt`: Sample training data containing a large sample of names.
- `README.md`: Project documentation.
## Getting Started


### Prerequisites


- **CUDA TOOLKIT**: Ensure you have CUDA TOOLKIT installed. The project was made using [CUDA Toolkit 12.6](https://developer.nvidia.com/cuda-downloads?target_os=Linux).
- **GCC C++17**: The project uses C++17 features, so you'll need to have GCC installed with C++17 support
- **UBUNTU-24**: The project was configured for Ubuntu on WSL, compatibility is not guaranteed on any other operating system.


### Installation


1. **Clone the repository**:


2. **Compile the project**:
   Compile the code using the makefile
   ```bash
   cd version_3
   make
   ```


3. **Run the project**:
   Run the compiled program with:
   ```bash
   ./compiledProgram.o
   ```


### Usage


The model is currently configured for lowercase characters only, changes to main will be needed to be made to expand this.


To run the data on a different dataset, first make sure it is in the same format as names.txt, then change the file name in main.cpp


To tune the hyperparameters, simply change their values in main.cpp, the purpose of these parameters are shown below:


### Hyper Parameters
- **CONTEXT_WINDOW**: The amount of characters back the model can "see".
- **LOOKUP_DIMENSIONS**: The Dimensionality of the embeddings.
- **HIDDEN_LAYER_SIZE**: The size of the hidden layer matrices.
- **NUM_EXAMPLES**: The batch size when training, lower values are faster but result in more noise.
- **NUM_ITER**: The number of training iterations to train the model.
- **LEARNING_RATE**: The learning rate, this rate is reduced when the model has done half of the training.

## Sample Outputs

Here is a sample of the output that the model generated with 20,000 training loops:
```
marson
brellin
chnoum
eyreltons
nigha
mioah
ramira
deree
lexh
havik
aezoeltuia
irosa
tori
komstee
ioreely
nerau
wilie
aychil
liyua
jbrelynn
ato
idetann
antiai
brynallien
ashmini
lavini
jaiandic
tesiyco
javcyn
loja
acnelra
losi
lwais
mautam
oliviha
```
Obviously some work needs to be done to improve the quality of these names

## Versions

- `Version 1`: Implements a custom GPU matrix multiplication function
- `Version 2`: Uses CUBLAS to do matrix multiplication
- `Version 3`: Completely reworks the structure to reduce memory rewrites for nodes

## Roadmap

- [ ] Implement a pool for the GPU memory
- [ ] Improve backwards and forwards pass efficiency by increasing parallelism
- [ ] Implement batch normalisation
- [ ] Implement structure similar to [Wavenet](https://arxiv.org/pdf/1609.03499)
- [ ] Experiment with CuDNN for potential efficiency gains


## Acknowledgments


Inspired by Andrej Karpathy's [makemore](https://github.com/karpathy/makemore), this project closely follows his implementation and his youtube series was crucial in my understanding of neural networks.


[Justin Johnson's Michigan Online lecture on backpropagation](https://www.youtube.com/watch?v=dB-u77Y5a6A&t=3353s) was crucial in my understanding of gradient calculations and his lecture helped me immensely.
