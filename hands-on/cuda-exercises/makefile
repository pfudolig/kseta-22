CC = @nvcc -std=c++11

all: ex1 ex2 ex3 ex4 ex5

ex1 : ex01/redo_cuda_mem_model.cu
	$(CC) -o ex1 ex01/redo_cuda_mem_model.cu
ex2 : ex02/redo_kernel.cu
	$(CC) -o ex2 ex02/redo_kernel.cu
ex3 : ex03/redo_matrix.cu
	$(CC) -o ex3  ex03/redo_matrix.cu
ex4 : ex04/ex04.cu
	$(CC) -o ex4 ex04/ex04.cu
ex5 : ex05/reduction.cu
	$(CC) -o ex5  ex05/reduction.cu 