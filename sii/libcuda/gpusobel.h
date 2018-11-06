#include "headers.h"

extern "C" cudaError_t StartKernel_Object_Detection(uchar3 *pArrayA, uchar* pArrayR,int cols,int rows);

void GPUSobel(cv::Mat& img, cv::Mat out) {
    uchar3* pPixel = img.ptr<uchar3>(0);
	int sizeImg = img.rows * img.cols;

	// Je crée un image de retour en grayscale (uchar)
	Mat imgRetour(img.rows,img.cols,CV_8U);
	uchar* pPixelRetour = imgRetour.ptr<uchar>(0);

	cudaError_t t = StartKernel_Object_Detection(pPixel,pPixelRetour,img.cols,img.rows);
	if(t != cudaSuccess) {
		std::cout << "Erreur durant l'execution du kernel" << std::endl;
		return;
	}
}