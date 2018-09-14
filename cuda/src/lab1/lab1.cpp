#include "../../../include/cvheaders.hpp"
#include "../../../include/cudaheaders.hpp"
#include <tclap/CmdLine.h>

#include <iostream>
#include <string>
#include <math.h>

using namespace std;
using namespace cv;


extern "C" cudaError_t StartKernel_RGB_TO_HSV(uchar3 *pArrayA,float3 *pArrayR,int size);


float maxx(float a,float b,float c) {
	float max;
	max = a > b ? a : b;
	max = max > c ? max : c;
    return max;
}

float minn(float a,float b,float c) {
	float min;
	min = a < b ? a : b;
	min = min < c ? min : c;
    return min;
}

void HsvToRgbCPU(uchar3* ArrayA,float3* ArrayR, int size) {
	for(int index=0;index<size;index++) {
  	  	float Bs = ArrayA[index].x/255.0;
  	  	float Gs = ArrayA[index].y/255.0;
  	  	float Rs = ArrayA[index].z/255.0;
		
	    float CMax = maxx(Bs,Gs,Rs);
	    float CMin = minn(Bs,Gs,Rs);


	    float Delta = CMax - CMin;

		float h, s, v;
		v = CMax;


	    if(Delta == 0) {
	        h = 0;
	    } else if(CMax == Rs) {
	        h = (Gs-Bs)/Delta;
	    } else if (CMax == Gs) {
	        h = 2 + (Bs-Rs)/Delta;
	    } else { // Bs
	        h = 4 + (Rs-Gs)/Delta;
	    }

		h *= 60.0;

		if(h < 0) {
			h += 360.0;
		}

	    if (CMax == 0) {
	        s = 0;
	    } else {
	        s = Delta/CMax;
	    }
		ArrayR[index] = make_float3(h,s,v);
	}
}

int sensitivity = 15;
int greenLow = 85;
int greenHigh = 143;

void FilterBackground(float3* ArrayA, uchar3* ArrayR,int size) {
	for(int index=0;index<size;index++) {
		float h = ArrayA[index].x;
		float v = ArrayA[index].z;
		if (h >= greenLow &&  h <= greenHigh && v > 0.5) {
			ArrayR[index] = make_uchar3(255,255,255);
		} else {
			ArrayR[index] = make_uchar3(0,0,0);
		}
	}
}

void HsvToRgb(Mat img) {
	uchar3* pPixel = img.ptr<uchar3>(0);
	int sizeImg = img.rows * img.cols;
	Mat imgRetour(img.rows,img.cols,CV_32FC3);

	printf("MatOriginal size is %d Cols : %d Rows : %d Channels : %d\n",img.size(),img.cols, img.rows,img.channels());
	printf("MatRetour size is %d Cols : %d Rows : %d Channels : %d\n",imgRetour.size(),imgRetour.cols, imgRetour.rows,imgRetour.channels());

	float3* pPixelR = imgRetour.ptr<float3>(0);
	HsvToRgbCPU(pPixel,pPixelR,sizeImg);
	
/*	cudaError_t t = StartKernel_RGB_TO_HSV(pPixel,pPixelR,sizeImg);
	if (t != cudaSuccess) {
		printf("GPU error %d : %s \n",t,cudaGetErrorString(t));
		return;
	}*/

	Mat imgFilter(img.rows,img.cols,img.type());
	uchar3 *pPixelFilter = imgFilter.ptr<uchar3>(0);

	FilterBackground(pPixelR,pPixelFilter,sizeImg);

	imshow("Filter",imgFilter);
	//imgRetour.convertTo(imgRetour, CV_8UC3,255.0);
	imshow("HSV image", imgRetour);




	Mat imgHsv;
	cvtColor(img,imgHsv,COLOR_BGR2HSV);
	imshow("HSV OpenCV",imgHsv);

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
