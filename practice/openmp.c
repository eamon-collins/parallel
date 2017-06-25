#include <stdio.h>
#include <stdlib.h>

#define length 1000

int main(int argc, char *argv[]){
	int size = length * sizeof(int);
	int *arr1 = malloc(size);
	int *arr2 = malloc(size);
	int *arr3 = malloc(size);

	int i;
	for(i=0; i < length; i++){
		arr1[i] = 7;
		arr2[i] = 5;
	}

	#pragma omp parallel for
	for(i=0; i < length; i++){
		arr3[i] = arr1[i] + arr2[i];
	}

	for(i=0; i < length; i++){
		printf("%d ", arr3[i])
	}

	return 0;

}