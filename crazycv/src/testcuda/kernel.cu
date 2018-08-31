
// Mon kernel cuda pour manipuler mes pixels par un scalaire
__global__ static void Kernel_ScalaireMulMat_Int(int *MatA, int k, int *MatR, dim3 DimMat) {
    int index = threadIdx.y * blockDim.x + threadIdx.x;
    MatR[index] = MatA[index] * k;
}

int BLOCK_SIZE = 1;

extern "C" cudaError_t StartKernel(int *pMatA, int k, int *pMatR, dim3 DimMat) {
    // Assure qu'on peut belle et bien utiliser cuda
    cudaError_t cudaStatus = cudaSetDevice(0);
    if(cudaStatus != cudaSuccess) {
        return cudaStatus;
    }
    //Crée nos pointeur utiliser par cuda
    int *MatA, *MatR;

    // Crée la dimension de mon block
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    // Crée la dimension de ma grille
    dim3 dimGrid(iDivUp(DimMat.x, BLOCK_SIZE), iDivUp(DimMat.y, BLOCK_SIZE));

    // Alloue l'espace mémoire des deux matrices sur le gpu
    // calcul de l'espace de notre array de pixel qui représente l'image
    size_t memSize = DimMat.x * DimMat.y * sizeof(int)
    cudaStatus = cudaMalloc( (void**)&MatA,memSize);
    if(cudaStatus != cudaSuccess){
        return cudaStatus;
    }
    cudaStatus = cudaMalloc( (void**)&MatR,memSize);
    if(cudaStatus != cudaSuccess) {
        return cudaStatus;
    }
    // Copie la matrice a dans la mémoire
    cudaMemcpy(MatA,pMatA, memSize, cudaMemcpyHostToDevice);

    // Démarre le kernel
    Kernel_ScalaireMulMat_Int<<<1,1>>>((int*)MatA,(int)k,(int*)MatR,DimMat);

    return cudaStatus;

}
