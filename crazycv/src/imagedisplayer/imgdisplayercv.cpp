#include "../../lib/capture.hpp"
#include "../../lib/file.hpp"
#include <tclap/CmdLine.h>


// width et height capturer par default
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;


int main(int argv,char ** argc)
{
	std::string fileName;
	bool displayInfo;
	bool stretchWindow;
	// parse les cmd line arguments pour savoir qu'elle photo show
	// le gros
	try {

		TCLAP::CmdLine cmd("Program pour affichier une image dans une fenetre", ' ',"0.9");

		// Definit un argument pour la photo a affichier
		TCLAP::ValueArg<std::string> fileArg("f","file","File to display",true,"","string");
		// Ajout cette arguments a la liste d'arguments
		cmd.add(fileArg);

		// Définit un argument pour affichier de l'info supplementaire sur l'image
		TCLAP::SwitchArg infoArg("i","info","Display information about the image",cmd,false);

		// Affiche la fenetre la grosseur de l'image
		TCLAP::SwitchArg sizeWindow("s","stretch","Strech window to be the size of the image",cmd,true);

		cmd.parse(argv,argc);

		// Get la value de chaque'un des arguments
		fileName = fileArg.getValue();
		if(fileName == "") {
			std::cerr << "The file name is empty." << std::endl;
			return 1;
		}
		displayInfo = infoArg.getValue();
		stretchWindow = sizeWindow.getValue();

	} catch(TCLAP::ArgException &e) {
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
		return 1;
	}

	// assure que le fichier existe et qu'il est de la bonne extension
	ValidFile(fileName,Extensions,1);

	Mat image = imread(fileName,CV_CAP_MODE_RGB);
	if(image.empty()) {
		printf("Image is empty");
		return 1;
	}

	// affiche de l'info sur l'image loader dans le terminal
	if (displayInfo) {
		Size size = image.size();
		printf("Image Name : %s Type : %s \n",fileName, GetFileExtensions(fileName));
		printf("Size Width : %d Height : %d\n",size.width,size.height);
		printf("There is %d Cols and %d Rows\n", image.cols,image.rows);
		printf("There is %d channels",image.channels());
		
	}

	String nameWindow = "Displaying  "+fileName;
	int modeWindow = 0;
	if(stretchWindow) {
		modeWindow = cv::WINDOW_AUTOSIZE;
	} 
	
	cv::namedWindow(nameWindow,modeWindow);			


	cv::imshow("Old img",image);

	printf("Press any key to close the window");
	
	cv::waitKey(0);
	
	return 0;
}

