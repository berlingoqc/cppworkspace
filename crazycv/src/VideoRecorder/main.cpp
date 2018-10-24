#include "../../../include/cvheaders.hpp"

/*
    Capture l'image d'un video du debut a la fin de l'execution du programme
    avec opencv

*/
using namespace std;
using namespace cv;


#define CAPTURE_HEIGHT 800;
#define CAPTURE_WIDTH 600;

std::string AppName = "Enregistrement video ";


int main(int argv,char ** argc)
{
    int         device = 0;
    std::string fileName = "recording.mp4";
    int         fps = 24;
    int         widthVideo = CAPTURE_WIDTH;
    int         heightVideo = CAPTURE_HEIGHT;

    // Recoit par mode commande l'a chande de connection a utiliser pour la camera
    // Recoit aussi la résolution et le nom du fichier out
    AppName = AppName.append(fileName);    

    

	cv::VideoCapture cap(device);
    if(!cap.isOpened()) {
        std::cerr << "Impossible d'ouvrir la capture sur le device " << device << std::endl;
        return 1;
    }

    cap.set(CAP_PROP_FRAME_HEIGHT,heightVideo);
    cap.set(CAP_PROP_FRAME_WIDTH,widthVideo);

    cv::namedWindow(AppName);

    cv::Mat img;

    Size frameSize(widthVideo,heightVideo);
    VideoWriter videoWriter(fileName,CV_FOURCC('X','2','6','4'),20,frameSize,true);
    if(!videoWriter.isOpened()) {
        std::cerr << "Impossible d'ouvrir le fichier " << fileName << " en écriture" << std::endl;
        return 1;
    }

    while(1) {
        if(!cap.read(img)) {
        	std::cerr << "Error capturing " << std::endl;
        	return 1;
        }


        videoWriter.write(img);
        imshow(AppName,img);

                
        if(cv::waitKey(1) > 0) {
        	// Quitte la boucle
    		break;
		}
	}
    
    // Clair les ressources a la fin du programme
    cap.release();
    videoWriter.release();
    destroyAllWindows();
    return 0;
}