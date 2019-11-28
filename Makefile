CC=/Developer/NVIDIA/CUDA-10.1/bin/nvcc

main: main.cu
	$(CC) -std=c++14 main.cu -o main