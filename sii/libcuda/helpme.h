#ifndef __HELPME_H__
#define __HELPME_H__
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"


// Typedef pour les uchar comme en c++
typedef unsigned char uchar;

// HandleError est une fonction qui affiche en détail un erreur cuda et l'emplacement 
static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

// HANDLE_ERROR definit pour englober automatiquement le fichier et la ligne ou l'erreur arrive
#define HANDLE_ERROR(err) { \
	HandleError(err,__FILE__,__LINE__);}

// HANDLE_NULL meme chose mais valide si le retourne est null
#define HANDLE_NULL(a) { \
    if (a == NULL) { printf("Host memory failed in %s at line %d\n", __FILE__,__LINE__);\
    exit(EXIT_FAILURE);}}


/*struct Threeshold_HSV {
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
*/
// Round le resultat de a / b a l'int superieur le plus pres
int iDivUp(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

// ValidPlateform valide que la plateform utilsé support belle et bien cuda
void ValidPlateform(bool printinfo) {
	int nbr;
	HandleError(cudaGetDeviceCount(&nbr),__FILE__,__LINE__);
	if(nbr < 0) {
		printf("No cuda device found...\n");
		exit(EXIT_FAILURE);
	}
	HandleError(cudaSetDevice(0),__FILE__,__LINE__);
	if(printinfo) {
		// Affiche de l'information a propos du device utilisé
	}
}

// Calcul le nombre de bloc 1D pour un array 1D 
int getBlockCount_1D_1D(int size,int blocksize) {
	int block_count = iDivUp(size,blocksize);
    printf("Starting cuda kernel with %d 1D Blocks and %d 1D Threads\r\n",block_count,blocksize);
	return block_count;
}


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

__device__ int getGlobalIDx_1D_1D() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int getGlobalIDx_1D_2D() {
	return blockIdx.x * blockDim.x * blockDim.y
		+ threadIdx.y * blockDim.x + threadIdx.x;
}

__global__ static void Kernel_RGB_TO_HSV(uchar3 *ArrayA,float3 *ArrayR,int size) {
    int index = getGlobalIDx_1D_1D();

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



#endif