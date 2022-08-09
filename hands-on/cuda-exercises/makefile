CC = @nvcc
GG = @gcc
#all: ex01 ex02 ex03 ex04 ex05

#all: ex01.o
#	$(GG) -o ex1 ex01.o

#ex01 : ex01/redo_cuda_mem_model.cu
#	${CC} -o ex1 ex01/redo_cuda_mem_model.cu
#ex02 : ex02/redo_kernel.cu
#	${CC} -o ex2 ex02/redo_kernel.cu

all: ex03 ex04 ex05

ex03 : ex03/redo_matrix.cu
	${CC} -o ex3  ex03/redo_matrix.cu
ex04 : ex04/ex04.cu
	${CC} -o ex4 ex04/ex04.cu
ex05 : ex05/reduction.cu
	${CC} -o ex5  ex05/reduction.cu 