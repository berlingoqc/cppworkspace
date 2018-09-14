#include "../../include/helpme.h"

#define BLOCK_SIZE  50


struct Threeshold_HSV {
    float3 lowerValue;
    float3 higherValue;

    __device__ bool InRange(float3 v) {
        if( v.x >= lowerValue.x && v.y >= lowerValue.y && v.z >= lowerValue.z) {
            
        }
    }

}


__global__ static void Kernel_Filter_BG_HSV(float3* arrayhsv,uchar3* arrayr,int size) {

}

extern "C" cudaError_t StartKernel_Object_Detection(uchar3 *pArrayA, uchar3* pArrayR, int size) {
    ValidPlateform(false);
    int BLOCK_COUNT = getBlockCount_1D_1D(size,BLOCK_SIZE);
    
    // Crée les pointeurs cuda pour nos images
    uchar3 *pArrayInitial;   // L'Array de mon image initial
    uchar3 *pArrayFilterBG;  // L'Array de mon image avec le bg filtrer
    uchar3 *pArraySobel;     // L'Array de mon image avec le filtre sobel

    float3 *pArrayHSV;       // L'array de mon image en hsv

    // Alloue l'espace mémoire pour changer mon image en hsv
    size_t memSize = size * sizeof(uchar3);
    size_t memSizeF = size * sizeof(float3);

    HANDLE_ERROR(cudaMalloc((void**)&pArrayInitial,memSize));
    HANDLE_ERROR(cudaMalloc((void**)&pArrayHSV,memSizeF));

    HANDLE_ERROR(cudaMemcpy(pArrayInitial,pArrayA,memSize,cudaMemcpyHostToDevice));

    // Démarre le kernel pour transformer l'image en HSV
    Kernel_RGB_TO_HSV<<<BLOCK_COUNT,BLOCK_SIZE>>>((uchar3*)pArrayInitial,(float3*)pArrayHSV,(int)size);


    // Pendant que sa process alloc le rest de ma mémoire pour les autres transformations
    
    // aloue la mémoire pour notre image filterbg
    HANDLE_ERROR(cudaMalloc((void**)pArrayFilterBG,memSize));
    
    

}


extern "C" cudaError_t StartKernel_RGB_TO_HSV(uchar3 *pArrayA,float3 *pArrayR,int size) {

    ValidPlateform(false);
    int BLOCK_COUNT = getBlockCount_1D_1D(size,BLOCK_SIZE);

    //Crée nos pointeur utiliser par cuda
    uchar3 *ArrayA;
    float3 *ArrayR;

    // Alloue l'espace mémoire des deux ArrayRices sur le gpu
    // calcul de l'espace de notre array de pixel qui représente l'image
    size_t memSize = size * sizeof(uchar3);
    cudaError_t cudaStatus = cudaMalloc( (void**)&ArrayA,memSize);
    
	if(cudaStatus != cudaSuccess){
        printf("Failed to allocate uchar3\n");
        return cudaStatus;
    }

    size_t memSizeF = size * sizeof(float3);
	cudaStatus = cudaMalloc( (void**)&ArrayR,memSizeF);
    if(cudaStatus != cudaSuccess) {
        printf("Failed to allocate float3\n");
        return cudaStatus;
    }
    
	// Copie l'array de donnnée vers le gpu
    cudaStatus = cudaMemcpy(ArrayA,pArrayA, memSize, cudaMemcpyHostToDevice);
    if(cudaStatus != cudaSuccess) {
        printf("Failt to copy ArrayA to GPU\n");
        return cudaStatus;
    }

    // Démarre le kernel
    Kernel_RGB_TO_HSV<<<BLOCK_COUNT,BLOCK_SIZE>>>((uchar3*)ArrayA,(float3*)ArrayR,(int)size);

    if(cudaDeviceSynchronize() == cudaSuccess) {
        printf("Finit d'execution du kernel\r\n");
    }

    // Fait une copie de l'array de resultat du gpu vers le cpu
    cudaStatus = cudaMemcpy(pArrayR,ArrayR,memSizeF,cudaMemcpyDeviceToHost);
    if(cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    cudaFree(ArrayA);
    cudaFree(ArrayR);


    return cudaSuccess;


}