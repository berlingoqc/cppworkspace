#include "../../include/helpme.h"


// Je suis un kerne qui multiplie les elements d'une liste par une scalaire pis qui les mets dans une autre liste
__global__ static void Kernel_ScalairArray_Int(int *ArrayA, int k, int *ArrayR,int size) {
    int index =  blockDim.x;
    if (index < size)
        ArrayR[index] = ArrayA[index] * k;
}

int BLOCK_SIZE = 1;
int THREAD_COUNT = 0;


extern "C" cudaError_t StartKernel_ScalairArray_Int(int *pArrayA, int k, int *pArrayR, int size) {
    // Assure qu'on peut belle et bien utiliser cuda
    cudaError_t cudaStatus = cudaSetDevice(0);
    if(cudaStatus != cudaSuccess) {
        return cudaStatus;
    }
    //Crée nos pointeur utiliser par cuda
    int *ArrayA, *ArrayR;
    // Alloue l'espace mémoire des deux ArrayRices sur le gpu
    // calcul de l'espace de notre array de pixel qui représente l'image
    size_t memSize = size * sizeof(int);
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
    Kernel_ScalairArray_Int<<<size,1>>>((int*)ArrayA,(int)k,(int*)ArrayR,(int)size);

    if(cudaDeviceSynchronize() == cudaSuccess) {
        printf("Finit d'execution du kernel");
    }

    // Fait une copie de l'array de resultat du gpu vers le cpu
    cudaStatus = cudaMemcpy(pArrayR,ArrayR,memSize,cudaMemcpyHostToDevice);
    if(cudaStatus != cudaSuccess) {
        return cudaStatus;
    }
    cudaFree(ArrayA);
    cudaFree(ArrayR);


    return cudaSuccess;

}
