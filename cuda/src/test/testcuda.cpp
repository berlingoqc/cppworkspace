#include "../../../include/cvheaders.hpp"
#include "../../../include/cudaheaders.hpp"
#include <tclap/CmdLine.h>

#include <iostream>
#include <string>
#include <math.h>

using namespace std;
using namespace cv;


int sobelX[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
int sobelY[3][3]={{-1,-2,-1},{0,0,0},{-1,0,1}};


const int ARRAY_SIZE = 300;

extern "C" cudaError_t StartKernel_ScalairArray_Int(uchar *pArrayA, int k, uchar *pArrayR, int size);
extern "C" cudaError_t StartKernel_SorelFiltre(uchar *pArrayA,uchar *pArrarR, int size);
extern "C" cudaError_t StartKernel_RGB_TO_HSV(uchar3 *pArrayA,float3 *pArrayR,int size);


void HsvToRgb(Mat img) {
	uchar3* pPixel = img.ptr<uchar3>(0);
	int sizeImg = img.rows * img.cols;

	Mat imgRetour = Mat::zeros(img.size(),CV_32F);
	float3* pPixelR = imgRetour.ptr<float3>(0);

	cudaError_t t = StartKernel_RGB_TO_HSV(pPixel,pPixelR,sizeImg);
	if (t != cudaSuccess) {
		printf("GPU error : %s \n",cudaGetErrorString(t));
	}

}

// TestPixelScalarCPU effectue la tache de pixel * scalair sur le cpu et affiche le temps d'execution
void TestPixelScalarCPU(Mat img,int scalar) {
	uchar* pPixel = img.ptr<uchar>(0);
	int sizeImg = img.rows * (img.cols);

	// Crée mon image de retour avec la meme grosseur et le meme type
	Mat imgRetour(img.size(), img.type());
	uchar* pPixelR = imgRetour.ptr<uchar>(0);

	for (int y = 0; y < sizeImg; ++y) {
		pPixelR[y] = pPixel[y]*scalar;
	}
	imshow("TestPixelScalarCPU", imgRetour);
}

// TestPixelScalarGPU effectue la tache de pixel * scalair sur le gpu et affiche le temps d'execution
void TestPixelScalarGPU(Mat img,int scalar) {
	// Get le pointeur du debut de l'image
	uchar* pPixel = img.ptr<uchar>(0);
	int sizeImg = img.rows * (img.cols);

	// Crée mon image de retour avec la meme grosseur et le meme type
	Mat imgRetour(img.size(), img.type());
	uchar* pPixelR = imgRetour.ptr<uchar>(0);

	
	cudaError_t t = StartKernel_ScalairArray_Int(pPixel, scalar, pPixelR, sizeImg);

	if (t == cudaSuccess) {
		// Successfully excute kernel
		imshow("TestPixelScalarGPU", imgRetour);
	}
	else {
		std::cerr << "Cuda error : " << cudaGetErrorString(t) << std::endl;
	}
}

void TestSorelCPU(Mat img) {
	uchar* pPixel = img.ptr<uchar>(0);
	int sizeImg = img.rows * (img.cols);

	// Crée mon image de retour avec la meme grosseur et le meme type
	Mat imgRetour(img.size(), img.type());
	for (int y=1;y<img.rows-1;y++) {
		for (int x=1;x<img.cols-1;x++) {
			// Get le 6 points nécessaire au tour
			int tl = img.at<uchar>(Point(x-1,y-1));
			int ml = img.at<uchar>(Point(x-1,y));
			int bl = img.at<uchar>(Point(x-1,y+1));

			int tr = img.at<uchar>(Point(x+1,y+1));
			int mr = img.at<uchar>(Point(x+1,y));
			int br = img.at<uchar>(Point(x+y,y+1));

			int tm = img.at<uchar>(Point(x,y-1));
			int bm = img.at<uchar>(Point(x,y+1));

			int gx = tl * sobelX[0][0] + ml * sobelX[1][0] + bl * sobelX[2][0] + tr * sobelX[0][2] + mr * sobelX[1][2] + br * sobelX[2][2];
			int gy = tl * sobelY[0][0] + tm * sobelY[0][1] + tr * sobelY[0][2] + bl * sobelY[2][0] + bm * sobelY[2][1] + br * sobelY[2][2];

			int g = sqrt(pow(gx,2)+pow(gy,2));
			if(g > 255) {
				g = g / 9;
			}

			imgRetour.at<uchar>(Point(x,y)) = g ; 
		}
	}

	imshow("TestSorelCPU",imgRetour);
	waitKey();

}


// TestSorelGPU démarre le kernel pour effectuer le filtre sorel sur l'image
void TestSorelGPU(Mat img) {
	// Pour appliquer mon filtre sorel je convertit l'image en float
	imshow("TestSorelGPU",img);
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

	Mat img = imread("test.jpg",IMREAD_GRAYSCALE);
	if (img.empty()) {
		std::cerr << "Failed to load image" << std::endl;
		return 1;
	}

	// Affiche la grandeur de l'image
	std::cout << "Width : " << img.cols << " Height : " << img.rows << std::endl;

	// Valide que l'image est seulement a un channels
	CV_Assert(img.depth() == CV_8U);

	/*
	std::cout << "Starting execution on CPU" << std::endl;
	TestPixelScalarCPU(img,3);

	std::cout << "Starting execution on GPU" << std::endl;
	TestPixelScalarGPU(img,3);
	*/	
	std::cout << "Starting preparation for Sorel filtre on GPU" << std::endl;
	TestSorelCPU(img);

	// Attend un key pour fermer le programme 
	if(waitKey() > 0) {
		std::cout << "Goodbye" << std::endl;
		return 0;
	}


	return 0;
}
