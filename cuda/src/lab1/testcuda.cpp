#include "../../../include/cvheaders.hpp"
#include "../../../include/cudaheaders.hpp"
#include <iostream>
#include <string>


using namespace std;

void ModifPixelByScalar(Mat& image,int scalar) {
    for(int y = 0; y < image.rows;++y) 
	    for( int x = 0; x < image.cols;++x)
	    {
		    // get le pixel
		    cv::Vec3b vec = image.at<cv::Vec3b>(Point(x,y));
		    // le multiplie par le scalar passé en fonction
	        image.at<cv::Vec3b>(Point(x,y)) = vec * scalar;
	            
        }
}

int ARRAY_SIZE = 300;

extern "C" cudaError_t StartKernel_ScalairArray_Int(int *pArrayA, int k, int *pArrayR, int size);


int main(int argv, char ** argc) {

    int *pArrayA,*pArrayR;
    int k = 3;

    pArrayA[ARRAY_SIZE];
    pArrayR[ARRAY_SIZE];

    for(int i = 0;i<ARRAY_SIZE;i++) {
        pArrayA[i] = i + 1;
    }


    cudaError_t t =  StartKernel_ScalairArray_Int(pArrayA,k,pArrayR,ARRAY_SIZE);
    if(t == cudaSuccess) {
        for(int i = 0;i<ARRAY_SIZE;i++) {
            printf("Element %d = %d",i,pArrayR[i]);
        }
        return 0;
    } else {
        printf("Echer a démarrer le kernel");
        return 1;
    }
    return 0;
    
}