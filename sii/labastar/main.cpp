/*
    Laboratoire A-Star

    Partit 1

    1. Construire l'image de la salle verte a l'aide de deux images prisent avec la camera du plafond
    2. Diviser l'image b en zone suffisamment grande pour le robot Youbot
    3. Supprimer le fond avec notre project cuda de debut de la session
    4. Determiner les zones occupées par un object

    Suggestion de Pan et Tilt :
        Coté couloir : PAN  = -41.99 Tilt = -70.0
        Coté G253    : PAN  = 134.69 Tilt = -70.0

    Partit 2

    1. Concevoir un algorithme de recherche de chemin de type AStart
    2. La solution doit pouvoir être monté étape par étape avec la touche 's'
    3. Toute comme dans la démo, chaque zone doit :
        i.  Afficher les informations suivantes à l’écran :
            1. Un numéro d’identification unique 
            2. Le score F,G et H
            3. Le nœud parent.
        ii. Être identifié par un Ptn bleu si elle fait partie de la liste ouverte.
        iii.Être identifié par un Ptn rouge si elle fait partie de la liste fermée.
        iv. Être identifié par un Ptn vert si elle fait partie du chemin trouvé (si chemin il y a).
*/
#ifdef _WITH_AXIS_COM
    #include "AxisCommunication.h"
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <thread>
#include <chrono>

#include <iostream>

using namespace cv;

// DÉFINITION 

#define YOUBOT_WIDTH_PX 200;
#define YOUBOT_HEIGHT_PX 200;

#define AXIS_CAM_IP "10.128.3.4"
#define AXIS_CAM_USER "etudiant"
#define AXIS_CAM_PW "gty970"


struct pathfind_info {
    Point   startingLocation;
    Point   endLocation;
    Size    objectSize;
};


// Variable qui indique si on utilse cuda ou sinon juste des fonctions opencv
bool useCUDA = false;
// Peut contenir le liens d'une image si oui utilise cette image au lieu
// d'obtenir l'image depuis AxisCommunication
std::string imageFromFilePath;
// Contient l'image du fichier précédent
cv::Mat imgNoAxis;

#ifdef _WITH_AXIS_COM
cv::VideoCapture vc;
CamData cam;
Axis axis(AXIS_CAM_IP, AXIS_CAM_USER, AXIS_CAM_PW);

cv::Mat getImage() {
    cv::Mat img1, img2;
    bool failed = true;
    while(failed) {
        failed = false;
        //axis.AbsolutePan(-161.934402f);
        axis.AbsolutePan(-41.99f);
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        axis.AbsoluteTilt(-70.0f);
        //axis.AbsoluteTilt(-66.159401f);
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        vc.read(img1);
        vc.read(img1);
        vc.read(img1);
		if (!vc.read(img1))
		{
			// Unable to retrieve frame from video stream
			std::cout << "Cannhot [sic] read image on Axis cam..." << std::endl << "Hit a key to try again." << std::endl;
			cv::waitKey(0);
			failed = true;
			vc.open("http://etudiant:gty970@10.128.3.4/axis-cgi/mjpg/video.cgi");
        }

        //axis.AbsolutePan(16.4405994f);
		axis.AbsolutePan(134.69f);
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
		axis.AbsoluteTilt(-70.0f);
        //axis.AbsoluteTilt(-70.701599f);
		std::this_thread::sleep_for(std::chrono::milliseconds(1500));
		vc.read(img2);
		vc.read(img2);
		vc.read(img2);
		if (!vc.read(img2))
		{
			// Unable to retrieve frame from video stream
			std::cout << "Cannhot [sic] read image on Axis cam..." << std::endl << "Hit a key to try again." << std::endl;
			cv::waitKey(0);
			failed = true;
			vc.open("http://etudiant:gty970@10.128.3.4/axis-cgi/mjpg/video.cgi");
        }
    }
    // Reconstruit nos deux images
    cv::Mat imgConcat;
    cv::flip(img2,img2,-1);
    cv::vconcat(img1, img2, imgConcat);
    cv::namedWindow("cat", CV_WINDOW_NORMAL);
	cv::imshow("cat", imgConcat);
    cv::waitKey(100);
    return imgConcat;
}

#else 
// Si on n'a pas axis cam on va loader une image depuis un fichier qui est passer en configuration
// ungly workaround
cv::Mat getImage() {
    return imgNoAxis;
}
#endif

// Utilise le backend opencv pour faire le traitement de l'image et get les contours
void processImageCV(const cv::Mat& img, cv::Mat& out) {
    cv::GaussianBlur(img,img, cv::Size(5,5), 2);
    cv::cvtColor(img,img, COLOR_BGR2HSV);
    inRange(img, cv::Scalar(50,0,0), cv::Scalar(83,255,128), out);
    //cv::morphologyEx(bin,bin, cv::MORPH_OPEN, cv::Mat::ones(7,7, CV_8UC1));
}


// Utilise mon backend cuda pour faire le traitement de l'image et get les contours
void processImageCUDA(const cv::Mat& origin, cv::Mat& out) {


}



void startMainLoop() {
	float vehicleWidth = YOUBOT_WIDTH_PX;
    #ifdef _WITH_AXIS_COM
        // Ouvre la video capture depuis la camera
	    vc.open("http://etudiant:gty970@10.128.3.4/axis-cgi/mjpg/video.cgi");
    #endif 

    std::vector<std::vector<cv::Point>> contours0;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::Mat bin;
    cv::Mat img;
	cv::Mat imgOrig = getImage();

    // Depuis l'image d'origin va chercher le robot cuca pour obtenir
    // sa largeur et sa longeur pour determiner la grosseur du grillage

    Size cucaSize(0,0);
    Point startingLocation(0,0);

    // Affiche le grillage sur l'image avec les résultats obtenue
    // et permet de cliquer pour choisir la zone de destination
    // si elle n'est pas deja fournit



    while (false) {
        // Passe notre image vers notre fonction cuda pour la traiter
        img = imgOrig.clone();

    }
    

    #ifdef _WITH_AXIS_COM
        vc.release();
    #endif
}


int main() {
    startMainLoop();
   return 0;
}