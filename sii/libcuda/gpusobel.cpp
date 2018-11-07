#include "gpusobel.h"

void GPUSobel(cv::Mat& img, cv::Mat& out) {
	uchar3* pPixel = img.ptr<uchar3>(0);
	int sizeImg = img.rows * img.cols;

	// Je crée un image de retour en grayscale (uchar)
	uchar* pPixelRetour = out.ptr<uchar>(0);

	cudaError_t t = StartKernel_Object_Detection(pPixel, pPixelRetour, img.cols, img.rows);
	if (t != cudaSuccess) {
		std::cout << "Erreur durant l'execution du kernel" << std::endl;
		return;
	}
}
