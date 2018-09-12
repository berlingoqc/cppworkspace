#include "../../include/helpme.h"

__device__ float max(float a,float b,float c) {
    if( a > b && a > c) return a;
    if (b > a && b > c) return b;
    return c;
}

__device__ float min(float a,float b,float c) {
    if (a < b && a < c) return a;
    if (b < a && b < c) return b;
    return c;
}


__global__ static void Kernel_RGB_TO_HSV(uchar3 *ArrayA,float3 *ArrayR,int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    float Bs = ArrayA[index].z/255.0;
    float Gs = ArrayA[index].y/255.0;
    float Rs = ArrayA[index].x/255.0;

    float CMax = max(Bs,Gs,Rs);
    float CMin = min(Bs,Gs,Rs);

    float Delta = CMax - CMin;

    float h;
    if(Delta == 0) {
        h = 0;
    } else if(CMax == Rs) {
        h = 60 * (int((Gs-Bs)/Delta)%6);
    } else if (CMax == Gs) {
        h = 60 * (((Bs-Rs)/Delta)+2);
    } else { // Bs
        h = 60 * (((Rs-Gs)/Delta)+4);
    }

    float s;
    if (CMax == 0) {
        s = 0;
    } else {
        s = Delta/CMax;
    }
    
    ArrayR[index] = make_float3(h,s,CMax);
}


extern "C" cudaError_t StartKernel_RGB_TO_HSV(uchar3 *pArrayA,float3 *pArrayR,int size) {

    int BLOCK_SIZE = 50;
    int BLOCK_COUNT = iDivUp(size,BLOCK_SIZE);
    printf("Starting cuda kernel with %d 1D Blocks and %d 1D Threads\r\n",BLOCK_COUNT,BLOCK_SIZE);

    // Assure qu'on peut belle et bien utiliser cuda
    cudaError_t cudaStatus = cudaSetDevice(0);
    if(cudaStatus != cudaSuccess) {
        return cudaStatus;
    }
    //Crée nos pointeur utiliser par cuda
    uchar3 *ArrayA;
    float3 *ArrayR;

    // Alloue l'espace mémoire des deux ArrayRices sur le gpu
    // calcul de l'espace de notre array de pixel qui représente l'image
    size_t memSize = size * sizeof(uchar3);
    cudaStatus = cudaMalloc( (void**)&ArrayA,memSize);
    
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