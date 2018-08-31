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

extern "C" cudaError_t StartKernel(int *pMatA, int k, int *pMatR, dim3 DimMat);


int main(int argv, char ** argc) {
     
    Mat img = imread("test.jpg",CV_32SC1);
    // Get la dimension de l'image dans un dim3
    if(img.empty()) {
        std::cerr << "Image is empty" << std::endl;
        return -1;
    }
    int *pMatA,*pMatR;
    int k = 3;



    dim3 DimMat(img.rows,img.cols);


    cudaError_t t =  StartKernel(pMatA,k,pMatR,DimMat);
    if(t == cudaSuccess) {
        printf("Démarrer le kernel avec succes");
        return 0;
    } else {
        printf("Echer a démarrer le kernel");
        return 1;
    }
    printf("Hello world\n");
    return 0;
    
    /*
    Mat img = imread("test.jpg");
    if(img.empty()) {
        std::cerr << "Image is empty" << std::endl;
        return -1;
    }

    ModifPixelByScalar(img,2);
    imshow("Show img",img);
    waitKey(0);
    */
}