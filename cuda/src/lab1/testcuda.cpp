#include "../../../include/cvheaders.hpp"
#include "../../../include/cudaheaders.hpp"
#include <iostream>
#include <string>


using namespace std;
using namespace cv;

void ModifPixelByScalar(Mat& image, int scalar) {
	for (int y = 0; y < image.rows; ++y)
		for (int x = 0; x < image.cols; ++x)
		{
			// get le pixel
			Vec3b vec = image.at<cv::Vec3b>(Point(x, y));
			// le multiplie par le scalar pass� en fonction
			image.at<Vec3b>(Point(x, y)) = vec * scalar;
		}
}

const int ARRAY_SIZE = 300;

extern "C" cudaError_t StartKernel_ScalairArray_Int(uchar *pArrayA, int k, uchar *pArrayR, int size);

void TestPixelScalarCPU() {
	Mat img = imread("D:\\test.jpg");
	if (img.empty()) {
		std::cerr << "Failed to load image" << std::endl;
		return;
	}
	ModifPixelByScalar(img, 3);
	imshow("Troplolol", img);
	waitKey();
}

void TestPixelScalarGPU() {
	Mat img = imread("D:\\test.jpg",IMREAD_GRAYSCALE);
	if (img.empty()) {
		std::cerr << "Failed to load image" << std::endl;
		return;
	}
	// Valide que l'image est seulement a un channels
	CV_Assert(img.depth() == CV_8U);
	// Get le pointeur du debut de l'image
	uchar* pPixel = img.ptr<uchar>(0);
	int sizeImg = img.rows * (img.cols);

	int i = pPixel[sizeImg-1];
	int f = pPixel[0];



	// Cr�e mon image de retour avec la meme grosseur et le meme type
	Mat imgRetour(img.size(), img.type());
	uchar* pPixelR = imgRetour.ptr<uchar>(0);

	
	cudaError_t t = StartKernel_ScalairArray_Int(pPixel, 2, pPixelR, sizeImg);
	if (t == cudaSuccess) {
		// Successfully excute kernel
		imshow("lol", imgRetour);
		waitKey();
	}
	else {
		std::cerr << "Cuda error : " << cudaGetErrorString(t) << std::endl;
	}


}


int main(int argv, char ** argc) {
	
	TestPixelScalarGPU();

	uchar pArrayA[ARRAY_SIZE], pArrayR[ARRAY_SIZE];
	int k = 3;


	for (int i = 0; i<ARRAY_SIZE; i++) {
		pArrayA[i] = i + 1;
	}


	cudaError_t t = StartKernel_ScalairArray_Int(pArrayA, k, pArrayR, ARRAY_SIZE);
	if (t == cudaSuccess) {
		for (int i = 0; i<ARRAY_SIZE; i++) {
			printf("Element %d = %d\r\n", i, pArrayR[i]);
		}
		return 0;
	}
	else {
		printf("Echer a d�marrer le kernel");
		return 1;
	}
	return 0;

}