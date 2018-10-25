#include "../../../include/cvheaders.hpp"

/*
    Laboratoire 5 : tracking object depuis camera

    Faire le suivit d'object entrant dans le champs de vision de la caméra :
        1. Afficher un carré autour des objects
        2. Dire où l'object a quitter l'écran ( haut, bas , gauche, droite )
        
*/
using namespace std;
using namespace cv;


#define MAX_NUM_OBJECTS 10

#define MIN_OBJECT_AREA 20*20
#define MAX_OBJECT_AREA CAPTURE_WIDTH*CAPTURE_HEIGHT/1.5



int CAPTURE_WIDTH = 1280;
int CAPTURE_HEIGHT =  720;
const char * AppName = "Lab5 Tracking";
const char * wOrigin = "Image d'origine";
const char * wHsv = "Image HSV";
const char * wThres = "Image Thresholded";
const char * wMorph = "Image Morphological";
const char * wTrack = "Configuration";


// initialise les maximums et minimum pour hsv
int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;


std::string intToString(int nmb) {
    std::stringstream ss;
    ss << nmb;
    return ss.str();
}

void on_trackbar(int t, void*) {

} 

void createMyTrackBar() {
    cv::namedWindow(wTrack,0);
    char TrackbarName[50];
    sprintf( TrackbarName, "H_MIN", H_MIN);
	sprintf( TrackbarName, "H_MAX", H_MAX);
	sprintf( TrackbarName, "S_MIN", S_MIN);
	sprintf( TrackbarName, "S_MAX", S_MAX);
	sprintf( TrackbarName, "V_MIN", V_MIN);
	sprintf( TrackbarName, "V_MAX", V_MAX);

    createTrackbar( "H_MIN", wTrack, &H_MIN, H_MAX, on_trackbar );
    createTrackbar( "H_MAX", wTrack, &H_MAX, H_MAX, on_trackbar );
    createTrackbar( "S_MIN", wTrack, &S_MIN, S_MAX, on_trackbar );
    createTrackbar( "S_MAX", wTrack, &S_MAX, S_MAX, on_trackbar );
    createTrackbar( "V_MIN", wTrack, &V_MIN, V_MAX, on_trackbar );
    createTrackbar( "V_MAX", wTrack, &V_MAX, V_MAX, on_trackbar );

}




bool parseKeyPress(int keyNum) {
    if(keyNum > 0)
        return true;
    return false;
}

void drawObject(int x, int y, cv::Mat& frame) {
    circle(frame,Point(x,y),20,Scalar(0,255,0),2);
    if(y-25>0)
        line(frame,Point(x,y),Point(x,y-25),Scalar(0,255,0),2);
    else 
        line(frame,Point(x,y),Point(x,0),Scalar(0,255,0),2);
    if(y+25<CAPTURE_HEIGHT)
        line(frame,Point(x,y),Point(x,y+25),Scalar(0,255,0),2);
    else 
        line(frame,Point(x,y),Point(x,CAPTURE_HEIGHT),Scalar(0,255,0),2);
    if(x-25>0)
        line(frame,Point(x,y),Point(x-25,y),Scalar(0,255,0),2);
    else 
        line(frame,Point(x,y),Point(0,y),Scalar(0,255,0),2);
    if(x+25<CAPTURE_WIDTH)
        line(frame,Point(x,y),Point(x+25,y),Scalar(0,255,0),2);
    else
        line(frame,Point(x,y),Point(CAPTURE_WIDTH,y),Scalar(0,255,0),2);

	putText(frame,intToString(x)+","+intToString(y),Point(x,y+30),1,1,Scalar(0,255,0),2);
}

// appliquer une errotion et une dilatation
void morphImg(cv::Mat& threshold) {
    // Crée les éléments pour appliquer une matrice de dilatation et d'erosion
    cv::Mat erodeElement = getStructuringElement(MORPH_RECT,Size(3,3));
    cv::Mat dilateElement = getStructuringElement(MORPH_RECT,Size(8,8));
    
    cv::erode(threshold,threshold,erodeElement);
    cv::erode(threshold,threshold,erodeElement);

    cv::dilate(threshold,threshold,dilateElement);
    cv::dilate(threshold,threshold,dilateElement);
}

void trackFilteredObject(int &x, int &y,cv::Mat threshold, cv::Mat& origin) {
    // copy d'image threshold dans l'image temporaire
    cv::Mat temp;
    threshold.copyTo(temp);

    // crée des vecteurs besoin pour trouver les outputs
    std::vector<std::vector<Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    // trouver les contours avec l'image filtrer
    cv::findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);

    // utilise la méthode pour detecter nos objects detecter
    double refArea = 0;
    bool objectFound = false;
    if(hierarchy.size() > 0) {
        int numObjects = hierarchy.size();
        if(numObjects<MAX_NUM_OBJECTS) {
            for(int index=0; index >= 0; index = hierarchy[index][0]) {
                Moments moment = moments((cv::Mat)contours[index]);
                double area = moment.m00;
                if(area > MIN_OBJECT_AREA && area < MAX_OBJECT_AREA && area>refArea) {
                    x = moment.m10/area;
                    y = moment.m01/area;
                    objectFound = true;
                    refArea = area;
                } else {
                    objectFound = false;
                }
            }
            if(objectFound == true) {
                putText(origin,"object détecter",Point(0,50),2,1,Scalar(0,255,0),2);
                drawObject(x,y,origin);
            }
        } else {
            putText(origin,"trop d'object a l'écran",Point(0,50),2,1,Scalar(0,0,255),2);
        }
    } else {
        putText(origin,"Erreur pas de hierarchy trouver",Point(0,50),2,1,Scalar(0,0,255),2);
    }

}


int main(int argv,char ** argc)
{
    // Numero de la camera utilisé
    int device = 0;

    bool trackObject = false;
    bool morphOps = false;

    int x=0, y=0;

	cv::VideoCapture cap("C:\\Users\\wq\\lol.mpg");
    if(!cap.isOpened()) {
        std::cerr << "Impossible d'ouvrir la capture sur le device " << device << std::endl;
        return 1;
    }

    cap.set(CAP_PROP_FRAME_HEIGHT,CAPTURE_HEIGHT);
    cap.set(CAP_PROP_FRAME_WIDTH,CAPTURE_WIDTH);

    cv::namedWindow(AppName);

    createMyTrackBar();

    cv::Mat img, imgHsv, imgThresh, imgMorph;

    CAPTURE_HEIGHT = img.rows;
    CAPTURE_WIDTH = img.cols;

    while(1) {
        if(!cap.read(img)) {
        	std::cerr << "Error capturing " << std::endl;
        	return 1;
        }


        // Fait notre traitement sur l'image 
        cv::medianBlur(img,img,5);
        // change l'image en hsv
        cvtColor(img,imgHsv,COLOR_BGR2HSV);
        
        // filtre l'image hsv selon les valeurs
        inRange(imgHsv,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),imgThresh);

        // performe une operation morphologique pour eliminer le bruit
        if(morphOps) {
            morphImg(imgThresh);
        }

        if(trackObject) {
            trackFilteredObject(x,y,imgThresh,img);
        }

        imshow(wThres,imgThresh);
        imshow(AppName,img);
        imshow(wHsv,imgHsv);


                
        if(parseKeyPress(cv::waitKey(30))) {
        	// Quitte la boucle
    		break;
		}
	}
    cap.release();
    cvDestroyAllWindows();

    return 0;
}