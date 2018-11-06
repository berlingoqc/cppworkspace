//#include <tclap/CmdLine.h>
#include "headers.h"
#include "gpusobel.h"

#include <fstream>
#include <iostream>
#include <string>
#include <math.h>

using namespace std;
using namespace cv;


extern "C" cudaError_t StartKernel_Object_Detection(uchar3 *pArrayA, uchar* pArrayR,int cols,int rows);

std::string outputName = "";

void HsvToRgb(Mat img) {
	cv::Mat out;
	GPUSobel(img,out);

	imshow("Image resultante", imgRetour);

	if(outputName != "") {
		std::cout << "Enregistrement du résultat dans " << outputName << std::endl;
		imwrite(outputName,imgRetour);
	}
	imshow("Image d'origine",img);

	// Attend un key pour fermer le programme 
	if(waitKey() > 0) {;
		return;
	}
}

int main(int argv, char ** argc) {
	std::string fileName;

	bool		useAxis;
	std::string ipCamera;
	std::string userCamera;
	std::string pwCamera;
	/*
	try {
		TCLAP::CmdLine cmd("Laboratoire 1 Système Industriel Intélligent",' ',"1.0");

		// Définit un argument pour le nom du fichier a traiter
		TCLAP::ValueArg<std::string> fileArg("f","file","Ficher à utiliser pour la transformation (test.jpg) par défault",false,"test.jpg","string");
		TCLAP::ValueArg<std::string> outputArg("o","output","Enregistre le résultat dans le fichier de ce nom",false,"","string");
		cmd.add(outputArg);
		cmd.add(fileArg);

		TCLAP::SwitchArg useCamera("c","camera","Utilise une camera Axis pour retrieve l'image a transformer",cmd,false);
		TCLAP::ValueArg<std::string> ipCam("i","ip","L'adresse ip de la camera axis",false,"127.0.0.1","string");
		TCLAP::ValueArg<std::string> userCam("u","user", "L'usager pour la camera axis",false,"","string");
		TCLAP::ValueArg<std::string> pwCam("p","password","Mot de passe de la camera",false,"","string");
		cmd.add(ipCam);
		cmd.add(userCam);
		cmd.add(pwCam);

		cmd.parse(argv,argc);
		fileName = fileArg.getValue();
		outputName = outputArg.getValue();
		useAxis = useCamera.getValue();

		if(useAxis) {
			ipCamera = ipCam.getValue();
			userCamera = userCam.getValue();
			pwCamera = pwCam.getValue();
		}
		

	} catch(TCLAP::ArgException &e) {
		std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
		return 1;
	}
	*/

	Mat img;

	// Si on utilise la camera pour retrieve l'image
	if(useAxis) {
		#ifdef AXIS_CAM
			#include "../../include/AxisCommunication.h"
			std::count << "Aquisition de l'image depuis la camera AXIS IP: " << ipCamera << " Usager : " << userCamera << std::endl;
			Axis axis(ipCamera.c_str(),userCamera.c_str(),pwCamera.c_str());
			
			if(!axis.GetImage(img)) {
				std::cerr << "Erreur d'acquision d'une image depuis la camera AXIS" << std::endl;
				exit(1);
			}
		#else
			// AXIS_CAM pas définit quand on n'a build le programme donc on peux pas faire ca
			std::cerr << "AXIS_CAM n'est pas définit, builder l'application avec l'option -DAXIS_CAM=1" << std::endl;
			exit(1);
		#endif

	} else {
	    std::cout << "Utilisation de l'image locale : " << fileName << std::endl;

		img = imread(fileName);

	}

	if (img.empty()) {
		std::cerr << "Erreur de lecture sur l'image" << std::endl;
		return 1;
	}

	// Affiche la grandeur de l'image
	std::cout << "Width : " << img.cols << " Height : " << img.rows << std::endl;

	HsvToRgb(img);

	return 0;
}
