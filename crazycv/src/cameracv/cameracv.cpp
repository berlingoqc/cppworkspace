#include "../../lib/capture.hpp"
#include "../../lib/file.hpp"
#include <tclap/CmdLine.h>

// width etdefineapturer par default
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;

int main(int argv,char ** argc)
{
	std::string fileName;

	int width;
	int height;

	int cameraIndex;

	// parse les cmd line arguments pour savoir qu'elle photo show
	// le gros
	try {

		TCLAP::CmdLine cmd("Program to capture image from the camera", ' ',"0.9");

		// Definit un argument pour la photo a affichier
		TCLAP::ValueArg<std::string> fileArg("n","name","Name of the picture to save, must be of a valid type",true,"","string");
		// Ajout cette arguments a la liste d'arguments
		cmd.add(fileArg);

		// Définit la résolution 
		TCLAP::ValueArg<int> widthArg("w","width","Width of the frame to capture picture",false,FRAME_WIDTH,"int");
		TCLAP::ValueArg<int> heightArg("e","hEight","Height of the frame to capture picture",false,FRAME_HEIGHT,"int");
		cmd.add(widthArg);
		cmd.add(heightArg);

		// Argument pour quelle camera utilisé index
		TCLAP::ValueArg<int> cameraArg("c","camera","Index of the camera to use",false,0,"int");
		cmd.add(cameraArg);

		// Affiche la fenetre la grosseur de l'image
		TCLAP::SwitchArg sizeWindow("s","stretch","Strech window to be the size of the image",cmd,true);

		cmd.parse(argv,argc);

		// Get la value de chaque'un des arguments
		fileName = fileArg.getValue();
		if(fileName == "") {
			std::cerr << "The file name is empty." << std::endl;
			return 1;
		}
		width = widthArg.getValue();
		height = heightArg.getValue();
		cameraIndex = cameraArg.getValue();

	} catch(TCLAP::ArgException &e) {
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
		return 1;
	}

	ImageWrapper*  img = new ImageWrapper();
	Resolution res = { width, height};
	if(!img->GetImageFromCamera(fileName,cameraIndex,res)) {
		std::cerr << "Could not get image from camera" << std::endl;
		return -1;
	}

	if(!img->Save(fileName)) {
		std::cerr << "Could not save file to disk " << fileName << std::endl;
		return 1;
	}



}