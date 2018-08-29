#include "../../lib/capture.hpp"
#include "../../lib/file.hpp"
#include <tclap/CmdLine.h>


Vec3b manipulateRedGreenBlue(Vec3b vec) {
// Fait la manipluation sur le pixel
	int r = vec[2];
	int v = vec[1];
	int b = vec[0];
	//printf("X : %d Y : %d R : %d G: %d B: %d\n",x,y,r,v,b);
	// Detection du rouge
	if (((r > 0 && r <= 10) || (r > 90 && r <= 255)) && b != 255 && v != 255) {
		vec = Vec3b(0,0,0);
	}

	return vec;
}



int main(int argv,char ** argc)
{
	std::string fileName;
	std::string outputFileName;

	int width = 1280;
	int height = 720;

	int cameraIndex;

	bool showOutput;

	// parse les cmd line arguments pour savoir qu'elle photo show
	// le gros
	try {

		TCLAP::CmdLine cmd("Program to capture image from the camera", ' ',"0.9");

		// Definit un argument pour la photo a affichier
		TCLAP::ValueArg<std::string> fileArg("f","name","To make the color switch on a existing file",false,"","string");
		TCLAP::ValueArg<std::string> outputFileArg("o","output","Name of the file to store the result",false,"","string");
		// Ajout cette arguments a la liste d'arguments
		cmd.add(outputFileArg);
		cmd.add(fileArg);

		// Définit la résolution 
		TCLAP::ValueArg<int> widthArg("w","width","Width of the frame to capture picture",false,width,"int");
		TCLAP::ValueArg<int> heightArg("e","hEight","Height of the frame to capture picture",false,height,"int");
		cmd.add(widthArg);
		cmd.add(heightArg);

		// Argument pour quelle camera utilisé index
		TCLAP::ValueArg<int> cameraArg("c","camera","Index of the camera to use",false,0,"int");
		cmd.add(cameraArg);

		TCLAP::SwitchArg showArg("s","show","Show the output of the transformation",cmd,false);

		cmd.parse(argv,argc);

		// Get la value de chaque'un des arguments
		fileName = fileArg.getValue();
		outputFileName = outputFileArg.getValue();
		width = widthArg.getValue();
		height = heightArg.getValue();
		cameraIndex = cameraArg.getValue();
		showOutput = showArg.getValue();

	} catch(TCLAP::ArgException &e) {
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
		return 1;
	}

	// si un fichier est entrer on le parse lui a lieu de la trans
	if(fileName != "") {
		ImageWrapper img(fileName);
		if(!img.Open()) {
			std::cerr << "could not open " << fileName << std::endl;
			return 1;
		}
		img.SetManipulatePixel(manipulateRedGreenBlue);
		img.TransformPixel();
		img.ShowAndWait();
	}

}