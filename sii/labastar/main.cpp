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
//#include "node.h"
//#include "2dhelp.h"

using namespace std::chrono;

// DÉFINITION 

#define YOUBOT_WIDTH_PX 200;
#define YOUBOT_HEIGHT_PX 200;

#define AXIS_CAM_IP "10.128.3.4"
#define AXIS_CAM_USER "etudiant"
#define AXIS_CAM_PW "gty970"



typedef std::vector<std::vector<cv::Point>> Contours;
typedef std::vector<cv::Vec4i> Hierarchy;

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
	cv::Sobel(out, out, 0, 1, 0, ksize = 5);
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
	cv::Sobel(out, out, 0, 1, 0, ksize = 5);
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

void getObjectInImage(cv::Mat&origin)
{
	cv::Mat bin(origin.rows, origin.cols, CV_8UC3);
	processImage(origin, bin);

	
}

std::promise<cv::Point> exitSignal;
std::future<cv::Point>  futureObj = exitSignal.get_future();



void mouse_callback(int even, int x, int y, int flags, void* user_data) {
	if (even == CV_EVENT_LBUTTONDOWN) {
		std::cout << "Left click at " << x << " " << y << std::endl;
		exitSignal.set_value(cv::Point(x,y));
		return;
	}
}


int distanceBetweenPoint(cv::Point p1, cv::Point p2) {
	int deltaX = p1.x - p2.x;
	int deltaY = p1.y - p2.y;
	return sqrt((deltaX*deltaX + deltaY * deltaY));
}

class Node {
	cv::Point	central_point;
	cv::Point	size;
	
	int	fcost, hcost, gcost;

	bool visited;
	
	Node*		parent;


public:
	Node(cv::Point central_point, cv::Point size, Node* parent = nullptr) {
		this->central_point = central_point;
		this->size = size;
		this->parent = parent;
		if(parent == nullptr) {
			fcost = 0;
			hcost = 0;
			gcost = 0;
		}
	}

	cv::Point getTopLeftCorner() {
		int pX;
		int pY;
		return {pX,pY};
	}

	cv::Point getCentralPoint(){
		return central_point;
	}

	int calculateCost(const Node* end, Node* potentiel_parent) {
		float g, h, f;
		h = distanceBetweenPoint(central_point, end->central_point);
		if (parent == nullptr) {
			g = 0;
		}
		else {
			g = potentiel_parent->getGCost() + distanceBetweenPoint(potentiel_parent->central_point, central_point);
		}

		f = g + h;
		if (parent == nullptr || (f < fcost)) {
			parent = potentiel_parent;
			hcost = h;
			fcost = f;
			gcost = g;
		}

		return fcost;
	}


	Node* getParent() {
		return parent;
	}

	int getHCost() {
		return hcost;
	}
	
	int getFCost() {
		return fcost;
	}

	int getGCost() {
		return gcost;
	}

	int getVisited() {
		return visited;
	}

	void setVisisted(bool v) {
		visited = v;
	}

};

enum map_iterator_state {
	CAN_NEXT,
	CANT_NEXT,
	FOUND_END
};

class Map {
	int					width_node_nbr;
	int					width_node_size;
	int					height_node_nbr;
	int					height_node_size;
	
	std::vector<Node*>	liste_ouverte;
	std::vector<Node*>  liste_fermer;

	Node*				start;
	Node*				end;

	Contours			contours;
	Hierarchy			hierarchy;

	
public:
	Map(int wnn, int wns, int hnn, int hns) {
		width_node_nbr = wnn;
		width_node_size = wns;
		height_node_nbr = hnn;
		height_node_size = hns;
	}

	void setObstacle(Contours contours, Hierarchy hierarchy) {
		this->contours = contours;
		this->hierarchy = hierarchy;
	}

	cv::Mat add_info_image(const cv::Mat in) { 
		cv::Mat ret = in.clone();
		for (auto item : liste_ouverte) {

		}

		for (auto item : liste_fermer) {
		
		}
	}

	void draw_node(cv::Mat& m, Node* node, bool from_open) {
		char text[50];
		cv::Point p;
		if (node->getParent() == nullptr) {
			p = cv::Point(-1, -1);
		}
		else {
			p = node->getParent()->getCentralPoint();
		}
		// Get le coin superieur gauche du node
		int nbr = sprintf(text, "%d,%d -> %d,%d", node->getCentralPoint().x, node->getCentralPoint().y, p.x, p.y);
		cv::putText(m,text,cv::Point())
		
	}

	void setTarget(Node* start, Node* end) {
		this->start = start;
		

		liste_ouverte.push_back(start);
		this->end = end;
	}

	Node* getSmallestCostOpenList() {
		Node* n = nullptr;
		for (auto i : liste_ouverte) {
			if (n == nullptr) {
				n = i;
				continue;
			}
			if (n->getFCost() > i->getFCost()) {
				n = i;
			}
		}
		return n;
	}

	Node* getNodeFromPoint(cv::Point p) {
		int iX = p.x / width_node_size;
		int iY = p.y / height_node_size;

		return new Node({ iX*width_node_size + width_node_size / 2,iY*height_node_size + height_node_size / 2 }, { width_node_size,height_node_size });
	}

	bool isThereObstacleInNode(const Node* node) {
		return false;
	}

	map_iterator_state iterate_next() {
		Node* smallest_open = getSmallestCostOpenList();
	}
		 
};


void startMainLoop() {
	float vehicleWidth = YOUBOT_WIDTH_PX;
    #ifdef _WITH_AXIS_COM
        // Ouvre la video capture depuis la camera
		//vc.open("http://etudiant:gty970@10.128.3.4/axis-cgi/mjpg/video.cgi");
    #endif 

    std::vector<std::vector<cv::Point>> contours0;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::Mat img;

	const char* wImage = "Image";
	const char* wContour = "Image binaire";

	cv::namedWindow(wImage, 1);
	cv::setMouseCallback(wImage, mouse_callback);

	imageFromFilePath = "test.jpg";

	cv::Mat imgOrig = getImage();

    cv::Mat bin(imgOrig.rows, imgOrig.cols, CV_8U);

	cv::Size cucaSize(200, 200);


	img = imgOrig.clone();

	// ajoute les lignes de séparation des cases a l'image
	int rows = imgOrig.rows;
	int cols = imgOrig.cols;

	int gapWidth = cols / (cols / cucaSize.width);
	int gapHeight = rows / (rows / cucaSize.height);

	int nbrX = 0;
	int nbrY = 0;

	for (int i = gapWidth; i < cols; i += gapWidth) {
		nbrX++;
		cv::line(img, cv::Point(i, 0), cv::Point(i, rows), (0, 0, 0));
	}

	for (int i = gapHeight; i < rows; i += gapHeight) {
		nbrY++;
		cv::line(img, cv::Point(0, i), cv::Point(cols, i), (0, 0, 0));
	}

	Map m(nbrX,gapWidth,nbrY,gapHeight);
	// Va chercher les informations de node des deux points entrer
	Node* startingNode;
	Node* endNode;
	
	imshow(wImage, img);

	for (int i = 0; i < 2;i++) {
		// attend avec notre variable conditionel
		while (futureObj.wait_for(30ms) == std::future_status::timeout) {
			imshow(wImage, img);
			int v = cv::waitKey(1);
			if (v == 27) {
				return;
			}
		}

		if (i == 0) {
			startingNode = m.getNodeFromPoint(futureObj.get());
			cv::circle(img, startingNode->getCentralPoint(), 5, (0, 0, 255), -1);
		}
		else if (i == 1) {
			endNode = m.getNodeFromPoint(futureObj.get());
			cv::circle(img, endNode->getCentralPoint(), 5, (0, 0, 255), -1);
		}
		
		exitSignal = std::promise<cv::Point>();
		futureObj = exitSignal.get_future();
	}
	imshow(wImage, img);
	std::cout << "Tout les points sont fournit " << std::endl;

	m.setTarget(startingNode, endNode);

	bool run = true;
	
	processImage(imgOrig, bin);
	// Devrait ajouter les contours de l'image a ma map

	while (run) {
        // Passe notre image vers notre fonction cuda pour la traiter
		map_iterator_state state = m.iterate_next();
		switch (state) {
		case CANT_NEXT:
			// A explorer toutes les possiblités et on ne peux attendre le node
			break;
		case FOUND_END:
			// Est arriver au node de fin avec le meilleur chemin
			break;
		case CAN_NEXT:
			// On n'est pas arriver et on peut continuer vers une autre iteration
		default:

			break;
		}
		// Affiche les informations de la liste_ouverte

		// Affiche les informations de la liste_fermer


		int v = cv::waitKey(1);
		if (v == 27) {
			break;
		}
    }
    

    #ifdef _WITH_AXIS_COM
        vc.release();
    #endif
}


int main() {
    startMainLoop();
   return 0;
}