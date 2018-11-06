#include "../../include/helpme.h"

// Je suis un kerne qui multiplie les elements d'une liste par une scalaire pis qui les mets dans une autre liste
__global__ static void Kernel_ScalairArray_Int(uchar *ArrayA, int k, uchar *ArrayR,int size) {
    int index =  blockIdx.x * blockDim.x + threadIdx.x;
    ArrayR[index] = ArrayA[index] * k;
}

__device__ int sobelX[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
__device__ int sobelY[3][3]={{-1,-2,-1},{0,0,0},{-1,0,1}};

// Je suis un kernel qui applique le filre de sobel
__global__ static void Kernel_SobelOperator(float *ArrayA, int k, float *ArrayR, int size,int rows,int cols) {
    // Faut je get les nombres autour de mon index pou
    int index = blockIdx.x * blockDim.x + threadIdx.x;
}


int BLOCK_SIZE = 50;

extern "C" cudaError_t StartKernel_ScalairArray_Int(uchar *pArrayA, int k, uchar *pArrayR, int size) {

	// Calcul le nombre de thread par bloc que j'ai besoin
    int BLOCK_COUNT = iDivUp(size,BLOCK_SIZE);

    printf("Starting cuda kernel with %d 1D Blocks and %d 1D Threads\r\n",BLOCK_COUNT,BLOCK_SIZE);


    // Assure qu'on peut belle et bien utiliser cuda
    cudaError_t cudaStatus = cudaSetDevice(0);
    if(cudaStatus != cudaSuccess) {
        return cudaStatus;
    }
    //Crée nos pointeur utiliser par cuda
    uchar *ArrayA, *ArrayR;

    // Alloue l'espace mémoire des deux ArrayRices sur le gpu
    // calcul de l'espace de notre array de pixel qui représente l'image
    size_t memSize = size * sizeof(uchar);
    cudaStatus = cudaMalloc( (void**)&ArrayA,memSize);
    
	if(cudaStatus != cudaSuccess){
        return cudaStatus;
    }
    
	cudaStatus = cudaMalloc( (void**)&ArrayR,memSize);
    if(cudaStatus != cudaSuccess) {
        return cudaStatus;
    }
    
	// Copie l'array de donnnée vers le gpu
    cudaStatus = cudaMemcpy(ArrayA,pArrayA, memSize, cudaMemcpyHostToDevice);
    if(cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    // Démarre le kernel
    Kernel_ScalairArray_Int<<<BLOCK_COUNT,BLOCK_SIZE>>>((uchar*)ArrayA,(int)k,(uchar*)ArrayR,(int)size);

    if(cudaDeviceSynchronize() == cudaSuccess) {
        printf("Finit d'execution du kernel\r\n");
    }

    // Fait une copie de l'array de resultat du gpu vers le cpu
    cudaStatus = cudaMemcpy(pArrayR,ArrayR,memSize,cudaMemcpyDeviceToHost);
    if(cudaStatus != cudaSuccess) {
        return cudaStatus;
    }
    cudaFree(ArrayA);
    cudaFree(ArrayR);


    return cudaSuccess;

}
