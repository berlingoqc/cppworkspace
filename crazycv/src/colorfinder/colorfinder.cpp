#include "../../include/capture.hpp"
#include <tcap/Cmdline.h>
/*
Ce programme consiste en un program qui va dectecter les valeurs de couleurs d'un point precis d'une image
dans les différents colorspace
*/

cv::Mat img;


void mouseCallback(int event,int x,int y,int flags, void *userdata) {
    if(event == EVENT_LBUTTONDOWN) {
        // Affiche les informations de couleurs sur le pixel a cette emplacement
    }
}

int main(int argv,char ** argc) {

    std::string fileName;

    try {

        TCAP::CmdLine cmd("Program to detect color from image", ' ', "1.0");

        TCLAP::ValueArg<std::string> fileArg("f","file","Name of the file to work with",true,"","string");
        cmd.add(fileArg);

        cmd.parse(argv,argc);

        fileName = fileArg.getValue();
        if(fileName == "") {
            std::cerr << "This file is empty : " << fileName << std::endl;
            return -1;
        }        
    } catch(TCLAP::ArgException &e) {
        std::cerr << "Error cmd line : " << e.error() << " for arg " << e.argId() << std::endl;
        return -1;
    }

    img = imread(fileName)
    if(img.empty()) {
        std::cerr << "Error loading the image" << std::endl;
        return 1;
    }




}