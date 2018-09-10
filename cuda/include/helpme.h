#include <stdio.h>
#include "cuda_runtime.h"


typedef unsigned char uchar;

static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		return;
		//exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR(err) (HandleError(err,__FILE__.__LINE__))

#define HANDLE_NULL(a) { \
    if (a == NULL) { printf("Host memory failed in %s at line %d\n", __FILE__,__LINE__);\
    exit(EXIT_FAILURE);}}

// Round le resultat de a / b a l'int superieur le plus pres
int iDivUp(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}
