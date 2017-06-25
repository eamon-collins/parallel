#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define length 1024
#define threads_per_block 128


int main(int argc, char **argv) {
	int size = length * sizeof(int);
	int *arr1 = malloc(size);
	int *arr2 = malloc(size);
	int *arr3 = malloc(size);
	int *d_arr1, *d_arr2, *d_arr3;

	cudaMalloc((void **)&d_arr1, size);
	cudaMalloc((void **)&d_arr2, size);
	cudaMalloc((void **)&d_arr3, size);

	int i;
	for(i=0; i < length; i++){
		arr1[i] = 7;
		arr2[i] = 5;
	}

	cudaMemcpy(d_arr1, arr1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_arr2, arr2, size, cudaMemcpyHostToDevice);

	add<<<4 , threads_per_block>>>(d_arr1, d_arr2, d_arr3, length);

	cudaMemcpy(arr3, d_arr3, size, cudaMemcpyDeviceToHost);

	for (i=0; i < length; i++){
		printf("%d ", arr3[i]);
	}

	free(arr1); free(arr2); free(arr3);
	cudaFree(d_arr3); cudaFree(d_arr2); cudaFree(d_arr1);

	return 0;
}

__global__ void add(int *arr1, int *arr2, int *arr3, int length){
	int i;
	cells_per_block = length / blockDim.x;

	//i+=128 because there are 128 threads/block
	for (i=blockId.x*cells_per_block; i < (blockId.x + 1)*cells_per_block; i+=threads_per_block){
		arr3[i + threadId.x] = arr1[i + threadId.x] + arr2[i + threadId.x];
	}
}