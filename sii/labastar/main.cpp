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
#ifdef _WITH_CUDA
	#include "gpusobel.h"
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <thread>
#include <future>
#include <chrono>

#include <iostream>


#include "node.h"
#include "map.h"

using namespace std::chrono;

// DÉFINITION 

#define YOUBOT_WIDTH_PX 200;
#define YOUBOT_HEIGHT_PX 200;

#define AXIS_CAM_IP "10.128.3.4"
#define AXIS_CAM_USER "etudiant"
#define AXIS_CAM_PW "gty970"
;

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
			std::cout << "Impossible de lire une image..." << std::endl << "Presser une touche pour recommencer." << std::endl;
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
			std::cout << "Impossible de lire une image..." << std::endl << "Presser une touche pour recommencer." << std::endl;
			cv::waitKey(0);
			failed = true;
			vc.open("http://etudiant:gty970@10.128.3.4/axis-cgi/mjpg/video.cgi");
        }
    }
    // Reconstruit nos deux images
    cv::Mat imgConcat;
    cv::flip(img2,img2,-1);
    cv::vconcat(img1, img2, imgConcat);
    //cv::namedWindow("cat", CV_WINDOW_NORMAL);
	//cv::imshow("cat", imgConcat);
    //cv::waitKey(100);
    return imgConcat;
}

#else 
// Si on n'a pas axis cam on va loader une image depuis un fichier qui est passer en configuration
// ungly workaround
cv::Mat getImage() {
	cv::Mat m = cv::imread(imageFromFilePath);
	if (!m.empty()) {
		std::cerr << "Impossible de charger l'image de remplacement " << std::endl;
	}
	return m;
}
#endif
#ifndef _WITH_CUDA
// Utilise le backend opencv pour faire le traitement de l'image et get les contours
void processImage(cv::Mat& img, cv::Mat& out) {
    //cv::GaussianBlur(img,img, cv::Size(5,5), 2);
    cv::cvtColor(img,img, cv::COLOR_BGR2HSV);
    inRange(img, cv::Scalar(50,0,0), cv::Scalar(83,255,128), out);

    //cv::morphologyEx(bin,bin, cv::MORPH_OPEN, cv::Mat::ones(7,7, CV_8UC1));
}

#else
// Utilise mon backend cuda pour faire le traitement de l'image et get les contours
void processImage(cv::Mat& origin, cv::Mat& out) {
	GPUSobel(origin, out);
}

#endif

std::promise<cv::Point> exitSignal;
std::future<cv::Point>  futureObj = exitSignal.get_future();

void mouse_callback(int even, int x, int y, int flags, void* user_data) {
	if (even == CV_EVENT_LBUTTONDOWN) {
		std::cout << "Left click at " << x << " " << y << std::endl;
		exitSignal.set_value(cv::Point(x,y));
	}
}

void mouse_callback_2(int even, int x, int y, int flags, void* user_data) {

}

void find_contours(const cv::Mat& imgOrig,Map& m) {
	std::vector<std::vector<cv::Point>> contours0;
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::Mat bin(imgOrig.rows, imgOrig.cols, CV_8U);
	cv::Mat img = imgOrig.clone();
	cv::GaussianBlur(imgOrig, img, cv::Size(7, 7), 2);

	processImage(img, bin);

	cv::findContours(bin, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	contours0.resize(contours.size());
	for (int k = 0; k < contours.size(); k++)
	{
		cv::approxPolyDP(cv::Mat(contours[k]), contours0[k], 6, false);

	}

	for (int k = 0; k < contours0.size(); k++)
	{
		if (contours0[k].size() > 2 && contours0[k].size() < 10)
		{
			for (cv::Point i : contours0[k])
			{
				Node* n = m.getNodeFromPoint(i);
				if (n != nullptr)
				{
					n->setHaveObstacle(true);
				}
			}
		}
	}

	cv::Mat newimg = imgOrig.clone();
	drawContours(newimg, contours0, -1, cv::Scalar(128, 255, 255), 3, cv::LINE_AA, hierarchy, 3);
	imshow("Line preview", newimg);
	imshow("Sobel output", bin);
}

void startMainLoop() {
	float vehicleWidth = YOUBOT_WIDTH_PX;
    #ifdef _WITH_AXIS_COM
        // Ouvre la video capture depuis la camera
		vc.open("http://etudiant:gty970@10.128.3.4/axis-cgi/mjpg/video.cgi");
    #endif 

    cv::Mat img;

	const char* wImage = "Image";
	const char* wContour = "Image binaire";


	imageFromFilePath = "test.jpg";

#ifdef _WITH_AXIS_COM
	std::cout << "Chargement de l'image depuis AxisCom" << std::endl;
#else
	std::cout << "Chargement de l'image depuis " << imageFromFilePath << std::endl;
#endif

	cv::Mat imgOrig = getImage();
	cv::resize(imgOrig, imgOrig, cv::Size(imgOrig.cols / 2, imgOrig.rows / 2));


#ifdef _WITH_AXIS_COM
	vc.release();
#endif


	cv::namedWindow(wImage, 1);
	cv::setMouseCallback(wImage, mouse_callback);

	cv::Size cucaSize(100, 100);



	Map m(cucaSize, imgOrig);

	find_contours(imgOrig,m);

	for (int i = 0; i < 2;i++) {
		// attend avec notre variable conditionel
		while (futureObj.wait_for(30ms) == std::future_status::timeout) {
			imshow(wImage, m.get_image_info());
			int v = cv::waitKey(1);
			if (v == 27) {
				return;
			}
		}

		if (i == 0) {
			Node* startingNode = m.getNodeFromPoint(futureObj.get());
			m.setStart(startingNode);
			//cv::circle(img, startingNode->getCentralPoint(), 5, (0, 0, 255), -1);
		}
		else if (i == 1) {
			Node* endNode = m.getNodeFromPoint(futureObj.get());
			m.setTarget(endNode);
			//cv::circle(img, endNode->getCentralPoint(), 5, (0, 0, 255), -1);
		}
		
		exitSignal = std::promise<cv::Point>();
		futureObj = exitSignal.get_future();
	}

	cv::setMouseCallback(wImage, mouse_callback_2);
	imshow(wImage, m.get_image_info());
	std::cout << "Tout les points sont fournit " << std::endl;


	bool run = true;

	// Devrait ajouter les contours de l'image a ma map
	cv::waitKey(0);

	while (run) {
        // Passe notre image vers notre fonction cuda pour la traiter
		map_iterator_state state = m.iterate_next();
		switch (state) {
		case CANT_NEXT:
			// A explorer toutes les possiblités et on ne peux attendre le node
			break;
		case FOUND_END:
			// Est arriver au node de fin avec le meilleur chemin
			std::cout << "Chemin trouver" << std::endl;
			run = false;
			break;
		case CAN_NEXT:
			// On n'est pas arriver et on peut continuer vers une autre iteration
		default:

			break;
		}


		cv::imshow(wImage, m.get_image_info());
		int v = cv::waitKey(0);
		if (v == 27) {
			break;
		}
    }
    

}


int main() {
    startMainLoop();
   return 0;
}