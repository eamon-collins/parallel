#include<stdio.h>
#include<sys/time.h>
#include<stdlib.h>
#include<iostream>
#include<math.h>

using namespace std;

//----------------------------------- Structures and Globals---------------------------------------------
#define tileDim 32 

typedef struct {
	int dimension1;
	int dimension2;	
} ArrayMetadata2D;

// metadata variables describing dimensionalities of all data structures involved in the computation
ArrayMetadata2D A_MD, B_MD, C_MD;
// pointers for input and output arrays in the host memory  
float *A, *B, *C, *C_CPU;
// pointers for input and output arrays in the device memory (NVIDIA DRAM)
float *A_GPU, *B_GPU, *C_GPU;
//dimension of each subarray (also called tiles.) All subarrays are square and equal in size. Should be the square root of threads per block for best occupancy

//----------------------------------- host function definitions -----------------------------------------

void allocateAndInitializeAB();
void computeCpuMMM();
void copyMatricesToGPU();
void copyResultFromGPU();
void compareHostAndGpuOutput();
void die(const char *error); 
void check_error(cudaError e);

//----------------------------------- CUDA function definitions -----------------------------------------
// A GPU kernel that computes the vector sum A + B
__global__ void matrix_mult(float *A, float *B, float *C, int dim) {
	
	int i, j;
	// determine the index of the thread among all GPU threads	
	int blockId = blockIdx.x;
	//I don't know if the grid info supplied by CUDA is in registers or what, but based on the sample code above
	//I'm explicitly placing them there just in case. I figure I have 64 registers per thread when using 1024 threads/block, might as well use them
	//int blockDim = blockDim.x;
	//int threadId = blockId * blockDim.x + threadIdx.x;
	int threadId = threadIdx.x;
	//int threadCount = gridDim.x * blockDim.x;
	//assumes multiplication of square matrices
	int totalDim = dim;
	int totalCells = totalDim * totalDim;
	//number of tiles in each row/column of the C matrix (and therefore the other two matrices)
	int tiles_per_dim =(totalDim / tileDim) +1;
	//index of the top left corner of the tile represented by this block in the original C array
	int tileCorner = (blockId / tiles_per_dim * tileDim * totalDim) + (blockId % tiles_per_dim * tileDim);
	//declaring the subarrays for tiles of A and B in shared memory
	__shared__ float s_A[tileDim*tileDim], s_B[tileDim*tileDim];
	//accumulator variable for the value this thread will place into it's designated spot in C
	float c_acc = 0.0f;
	//offset for the cell transferred to shared memory by this thread navigating from the current tile corner (which is calculated in loop)
	//differences between the matrices here (and in the corner calculation) are intended to get A by rows and B by columns such that
	//B is effectively transposed tile by tile
	int a_off = (threadId / tileDim) * totalDim + (threadId % tileDim);
	int b_off = (threadId % tileDim) * totalDim + threadId / tileDim; 
	//rows of each tile that the thread in question will be responsible for multiplying together
	//this ensures that each thread has the correct unique combination of one row from a and one row from b
	int a_row = threadId / tileDim;
	int b_row = threadId % tileDim;
	for(i = 0; i < tiles_per_dim; i++){
		//top left corners for the current operating tile of A and B arrays
		int Acorner = (blockId / tiles_per_dim) * tileDim * totalDim + i * tileDim;
		int Bcorner = i * tileDim * totalDim + blockId % tiles_per_dim * tileDim;
		//detects if the current threads responsibility is on the grid, ie not in the part of a tile that extends past the boundaries
		//of the matrix  threadId
		int a_ongrid = (((i != tiles_per_dim-1) || (threadId % tileDim < totalDim % tileDim)) && ((Acorner + a_off) < totalCells)) ? 1 : 0;
		int b_ongrid = (((blockId % tiles_per_dim != tiles_per_dim-1) || (threadId < (totalDim % tileDim) * tileDim)) && ((Bcorner + b_off) < totalCells)) ? 1 : 0;
	//	printf("%d:  %d + %d  a_ongrid: %d\n",blockId, Acorner, a_off, a_ongrid);	
		//tiles on the edge will go over the bounds of the matrix a small amount, in cells where that happens
		//I simply set them equal to 0 so they don't affect the computation
		if(a_ongrid)
			s_A[threadId] = A[Acorner + a_off];
		else
			s_A[threadId] = 0.0f;
		if(b_ongrid)
			s_B[threadId] = B[Bcorner + b_off];
		else
			s_B[threadId] = 0.0f;
		
		//synchronize so that no thread is operating on the tiles before they are properly initialized
		__syncthreads();
		for(j = 0; j < tileDim; j++){
			c_acc += (s_A[a_row*tileDim + j] * s_B[b_row*tileDim + j]);
		}
		//sync so that one warp doesn't start changing the data as other warps are using it to calculate
		__syncthreads();
	}
	//set the correct cell in C to be equal to the accumulated value
	//when using the correct tileCorner according to the blockId, a_off will actually also be the correct C offset.
	//if statement because it is possible this is one of the tiles that extend past the border of the matrix
	if((tileCorner + a_off < totalCells) && ((blockId % tiles_per_dim != tiles_per_dim-1) || (threadId % tileDim < totalDim % tileDim)))
		C[tileCorner + a_off] = c_acc;
}


//-------------------------------------------------------------------------------------------------------
int main(int argc, char **argv) {
	
	A_MD.dimension1 = (argc > 1) ? atoi(argv[1]) : 100;
	A_MD.dimension2 = (argc > 2) ? atoi(argv[2]) : A_MD.dimension1;
	B_MD.dimension1 = (argc > 3) ? atoi(argv[3]) : A_MD.dimension2;
	B_MD.dimension2 = (argc > 4) ? atoi(argv[4]) : B_MD.dimension1;
	C_MD.dimension1 = A_MD.dimension1;
	C_MD.dimension2 = B_MD.dimension2;

	printf("Matrix A is %d-by-%d\n", A_MD.dimension1, A_MD.dimension2);
	printf("Matrix B is %d-by-%d\n", B_MD.dimension1, B_MD.dimension2);
	printf("Matrix C is %d-by-%d\n", C_MD.dimension1, C_MD.dimension2);

	allocateAndInitializeAB();
	
	// matrix matrix multiplication in the CPU
	double elapsed;
/**	clock_t start = clock();	
	computeCpuMMM();
	clock_t end = clock();
        elapsed = (end - start) / (double) CLOCKS_PER_SEC;
        printf("Computation time in the CPU: %f seconds\n", elapsed);
	fflush(stdout);
**/
	copyMatricesToGPU();
	
	int threads_per_block = tileDim*tileDim;
	int num_blocks = ((C_MD.dimension1 / tileDim) + 1) * ((C_MD.dimension1 / tileDim) + 1);
	clock_t gstart = clock();
	matrix_mult<<<num_blocks, threads_per_block>>>(A_GPU, B_GPU, C_GPU, C_MD.dimension1);
	cudaThreadSynchronize();
	clock_t gend = clock();
	elapsed = (gend - gstart) / (double) CLOCKS_PER_SEC;
        printf("Computation time in the GPU: %f seconds\n", elapsed);

	copyResultFromGPU();
//	compareHostAndGpuOutput();
	return 0;
}

// allocate and initialize A and B using a random number generator
void allocateAndInitializeAB() {
	
	size_t sizeofA = A_MD.dimension1 * A_MD.dimension2 * sizeof(float);
	A = (float*) malloc(sizeofA);
	
	srand(time(NULL));
  	for (int i = 0; i < A_MD.dimension1; i++) {
		for (int j = 0; j < A_MD.dimension2; j++) {
			int index = i * A_MD.dimension2 + j;
			A[index] = (rand() % 1000) * 0.001; 
		}
	}
	
	size_t sizeofB = B_MD.dimension1 * B_MD.dimension2 * sizeof(float);
	B = (float*) malloc(sizeofB);
  	for (int i = 0; i < B_MD.dimension1; i++) {
		for (int j = 0; j < B_MD.dimension2; j++) {
			int index = i * B_MD.dimension2 + j;
			B[index] = (rand() % 1000) * 0.001; 
		}
	}
}

// allocate memory in the GPU for all matrices, and copy A and B content from the host CPU memory to the GPU memory
void copyMatricesToGPU() {
	
	size_t sizeofA = A_MD.dimension1 * A_MD.dimension2 * sizeof(float);
	check_error(cudaMalloc((void **) &A_GPU, sizeofA));
	check_error(cudaMemcpy(A_GPU, A, sizeofA, cudaMemcpyHostToDevice));
	
	size_t sizeofB = B_MD.dimension1 * B_MD.dimension2 * sizeof(float);
	check_error(cudaMalloc((void **) &B_GPU, sizeofB));
	check_error(cudaMemcpy(B_GPU, B, sizeofB, cudaMemcpyHostToDevice));
	
	size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
	check_error(cudaMalloc((void **) &C_GPU, sizeofC));
}

// copy results from C_GPU which is in GPU card memory to C_CPU which is in the host CPU for result comparison
void copyResultFromGPU() {
	size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
	C_CPU = (float*) malloc(sizeofC);
	check_error(cudaMemcpy(C_CPU, C_GPU, sizeofC, cudaMemcpyDeviceToHost));
}

// do a straightforward matrix-matrix multiplication in the CPU
// notice that this implementation can be massively improved in the CPU by doing proper cache blocking but we are
// not providing you the efficient CPU implementation as that reveals too much about the ideal GPU implementation
void computeCpuMMM() {
	
	// allocate the result matrix for the CPU computation
	size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
	C = (float*) malloc(sizeofC);
	
	// compute C[i][j] as the sum of A[i][k] * B[k][j] for all columns k of A
	for (int i = 0; i < A_MD.dimension1; i++) {
		int a_i = i * A_MD.dimension2;
		int c_i = i * C_MD.dimension2;
		for (int j = 0; j < B_MD.dimension2; j++) {
			int c_index = c_i + j;
			C[c_index] = 0;
			for (int k = 0; k < B_MD.dimension1; k++) {
				int a_index = a_i + k;
				int b_index = k * B_MD.dimension2 + j;
				C[c_index] += A[a_index] * B[b_index];
			}
		}
	}
}

// function to determine if the GPU computation is done correctly by comparing the output from the GPU with that
// from the CPU
void compareHostAndGpuOutput() {
	int totalElements = C_MD.dimension1 * C_MD.dimension2;
	int missmatchCount = 0;
	for (int i = 0; i < totalElements; i++) {
		if (fabs(C[i] - C_CPU[i]) > 0.01) {
			missmatchCount++;
			printf("mismatch at index %i: %f\t%f\n", i, C[i], C_CPU[i]);
		}
	}
	if (missmatchCount > 0) {
		printf("Computation is incorrect: outputs do not match in %d indexes\n", missmatchCount);
	} else {
		printf("Computation is correct: CPU and GPU outputs match\n");
	}
}

// Prints the specified error message and then exits
void die(const char *error) {
        printf("%s", error);
        exit(1);
}

// If the specified error code refers to a real error, report it and quit the program
void check_error(cudaError e) {
        if (e != cudaSuccess) {
                printf("\nCUDA error: %s\n", cudaGetErrorString(e));
                exit(1);
        }
}

