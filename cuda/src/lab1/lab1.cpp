#include "../../../include/cvheaders.hpp"
#include "../../../include/cudaheaders.hpp"
#include <tclap/CmdLine.h>

#include <iostream>
#include <string>
#include <math.h>

using namespace std;
using namespace cv;


extern "C" cudaError_t StartKernel_RGB_TO_HSV(uchar3 *pArrayA,float3 *pArrayR,int size);
extern "C" cudaError_t StartKernel_Object_Detection(uchar3 *pArrayA, uchar* pArrayR,int cols,int rows);

void HsvToRgb(Mat img) {
	uchar3* pPixel = img.ptr<uchar3>(0);
	int sizeImg = img.rows * img.cols;

	// Je crée un image de retour en grayscale (uchar)
	Mat imgRetour(img.rows,img.cols,CV_8U);
	uchar* pPixelRetour = imgRetour.ptr<uchar>(0);

	cudaError_t t = StartKernel_Object_Detection(pPixel,pPixelRetour,img.cols,img.rows);
	if(t != cudaSuccess) {
		std::cout << "Error during kernel execution" << std::endl;
	}

	imshow("HSV image", imgRetour);

	imwrite("output_sorel.jpg",imgRetour);

	imshow("Original",img);

	// Attend un key pour fermer le programme 
	if(waitKey() > 0) {;
		std::cout << "Goodbye" << std::endl;
		return;
	}
}

int main(int argv, char ** argc) {
	std::string fileName;
	try {
		TCLAP::CmdLine cmd("Laboratoire 1 Système Industriel Intélligent",' ',"1.0");

		// Définit un argument pour le nom du fichier a traiter
		TCLAP::ValueArg<std::string> fileArg("f","file","File to use for transformation",false,"test.jpg","string");
		cmd.add(fileArg);

		cmd.parse(argv,argc);
		fileName = fileArg.getValue();

	} catch(TCLAP::ArgException &e) {
		std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
		return 1;
	}


    std::cout << "Starting with image : " << fileName << std::endl;

	Mat img = imread(fileName);
	if (img.empty()) {
		std::cerr << "Failed to load image" << std::endl;
		return 1;
	}

	// Affiche la grandeur de l'image
	std::cout << "Width : " << img.cols << " Height : " << img.rows << std::endl;

	HsvToRgb(img);


	return 0;
}
