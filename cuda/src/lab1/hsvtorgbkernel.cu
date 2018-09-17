#include "../../include/helpme.h"

#define BLOCK_SIZE  50


struct Threeshold_HSV {
    float3 lowerValue;
    float3 higherValue;

    __device__ bool InRange(float3 v) {
        if( v.x >= lowerValue.x && v.y >= lowerValue.y && v.z >= lowerValue.z
            && v.x <= higherValue.x && v.y <= higherValue.y && v.z <= higherValue.z) {
            return true;
        }
        return false;
    }
};


__device__ int sobelX[3][3] {{-1,0,1},{-2,0,2},{-1,0,1}};
__device__ int sobelY[3][3] {{1,2,1},{0,0,0},{-1,-2,-1}};


// Doivent être initialiser avant de démarrer la function qui leur fait reference
__constant__ float3 lowerValue;
__constant__ float3 higherValue;


__device__ bool InRange(float3 v) {
    if( v.x >= lowerValue.x && v.y >= lowerValue.y && v.z >= lowerValue.z
        && v.x <= higherValue.x && v.y <= higherValue.y && v.z <= higherValue.z) {
        return true;
    }
    return false;
}

__global__ static void Kernel_Sobel_Operator(uchar* ArrayA, uchar* ArrayR, int cols, int rows) {
    int index = getGlobalIDx_1D_1D();
    int mtindex = index - cols;
    int mbindex = index + cols; // Middle Bottom
    // Si on n'est sur les board on fait rien
    if(mtindex < 0 || mbindex > (rows*cols)) {
	    return;
	}

    int lt = ArrayA[mtindex - 1];       // Left Top
    int rt = ArrayA[mtindex + 1];       // Right Top
    int lb = ArrayA[mbindex - 1];       // Left Bottom
    int rb = ArrayA[mbindex + 1];       // Right Bottom
    int ml = ArrayA[index - 1];    // MiddleLeft
    int mr = ArrayA[index + 1];    // MiddleRight
	int mt = ArrayA[mtindex];
	int mb = ArrayA[mbindex];

	int x_weight = lt * sobelX[0][0] + ml * sobelX[1][0] + lb * sobelX[2][0] + rt * sobelX[0][2] + mr * sobelX[1][2] + rb * sobelX[2][2];
    int y_weight = lt * sobelY[0][0] + mt * sobelY[0][1] + rt * sobelY[0][2] + lb * sobelY[2][0] + mb * sobelY[2][1] + rb * sobelY[2][2];

    x_weight = x_weight * x_weight;
    y_weight = y_weight * y_weight;

    float val = sqrtf(x_weight+y_weight+0.0);
    if(val > 255) {
        val = 255;
    }
    ArrayR[index] = (uchar)val;
}


__global__ static void Kernel_Filter_HSV_TO_GS(float3* ArrayA,uchar* ArrayR,int size) {
    int index =  getGlobalIDx_1D_1D();
	if (InRange(ArrayA[index])) {
		ArrayR[index] = 255;
        return;
    }
	ArrayR[index] = 0;
}

extern "C" cudaError_t StartKernel_Object_Detection(uchar3 *pArrayA, uchar* pArrayR, int cols,int rows) {
    ValidPlateform(false);
    int size = cols * rows;
    int BLOCK_COUNT = getBlockCount_1D_1D(size,BLOCK_SIZE);
    // Crée les pointeurs cuda pour nos images
    uchar3 *pArrayInitial;   // L'Array de mon image initial
    float3 *pArrayHSV;       // L'array de mon image en hsv
    uchar *pArrayFilterBG;  // L'Array de mon image avec le bg filtrer
    uchar *pArraySobel;     // L'Array de mon image avec le filtre sobel


    
    // Alloue l'espace mémoire pour changer mon image en hsv
    size_t memSize = size * sizeof(uchar3);
    size_t memSizeF = size * sizeof(float3);
    size_t memSizeU = size * sizeof(uchar);

    HANDLE_ERROR(cudaMalloc((void**)&pArrayInitial,memSize));
    HANDLE_ERROR(cudaMalloc((void**)&pArrayHSV,memSizeF));

    HANDLE_ERROR(cudaMemcpy(pArrayInitial,pArrayA,memSize,cudaMemcpyHostToDevice));

    // Démarre le kernel pour transformer l'image en HSV
    Kernel_RGB_TO_HSV<<<BLOCK_COUNT,BLOCK_SIZE>>>((uchar3*)pArrayInitial,(float3*)pArrayHSV,(int)size);

    // Pendant que sa process alloc le rest de ma mémoire pour les autres transformations
    
    // aloue la mémoire pour notre image filterbg
    HANDLE_ERROR(cudaMalloc((void**)&pArrayFilterBG,memSizeU));


    float3 lower = make_float3(85,0.0,0.5);
    float3 higher = make_float3(143,1.0,1.0);

    HANDLE_ERROR(cudaMemcpyToSymbol(lowerValue,&lower,sizeof(float3)));
    HANDLE_ERROR(cudaMemcpyToSymbol(higherValue,&higher,sizeof(float3)));

    // attend que la dernier kernel lancer soit finit
    HANDLE_ERROR(cudaDeviceSynchronize());

    printf("Fin execution conversion vers HSV\n");

    Kernel_Filter_HSV_TO_GS<<<BLOCK_COUNT,BLOCK_SIZE>>>((float3*)pArrayHSV,(uchar*)pArrayFilterBG,(int)size);

    HANDLE_ERROR(cudaMalloc((void**)&pArraySobel,memSizeU));

    HANDLE_ERROR(cudaDeviceSynchronize());

    Kernel_Sobel_Operator<<<BLOCK_COUNT,BLOCK_SIZE>>>((uchar*)pArrayFilterBG,(uchar*)pArraySobel,(int)cols,(int)rows);


    HANDLE_ERROR(cudaDeviceSynchronize());

    // Copie notre image final vers le cpu
    HANDLE_ERROR(cudaMemcpy(pArrayR,pArraySobel,memSizeU, cudaMemcpyDeviceToHost));

    cudaFree(pArrayInitial);
    cudaFree(pArrayHSV);
    cudaFree(pArrayFilterBG);    
    cudaFree(pArraySobel);
    
    return cudaSuccess;

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