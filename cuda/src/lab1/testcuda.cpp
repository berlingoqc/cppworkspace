#include "../../../include/cvheaders.hpp"
#include "../../../include/cudaheaders.hpp"
#include <tclap/CmdLine.h>

#include <iostream>
#include <string>


using namespace std;
using namespace cv;


const int ARRAY_SIZE = 300;

extern "C" cudaError_t StartKernel_ScalairArray_Int(uchar *pArrayA, int k, uchar *pArrayR, int size);
extern "C" cudaError_t StartKernel_SorelFiltre(uchar *pArrayA,uchar *pArrarR, int size);

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


	std::cout << "Starting execution on CPU" << std::endl;
	TestPixelScalarCPU(img,3);

	std::cout << "Starting execution on GPU" << std::endl;
	TestPixelScalarGPU(img,3);
	
	std::cout << "Starting preparation for Sorel filtre on GPU" << std::endl;
	TestSorelGPU(img);

	// Attend un key pour fermer le programme 
	if(waitKey() > 0) {
		std::cout << "Goodbye" << std::endl;
		return 0;
	}


	return 0;
}
