#include "starttracking.h"
#include "ui_starttracking.h"
#include "cvheaders.h"

#include <QFileDialog>
#include <QStandardPaths>
#include <QDesktopServices>

using namespace std;
using namespace cv;

#define MAX_NUM_OBJECTS 10

#define MIN_OBJECT_AREA 20*20




static const char * AppName = "Lab5 Tracking";
static const char * wOrigin = "Image d'origine";
static const char * wHsv = "Image HSV";
static const char * wThres = "Image Thresholded";
static const char * wMorph = "Image Morphological";
static const char * wTrack = "Configuration";


// initialise les maximums et minimum pour hsv
int H_MIN = 170;
int H_MAX = 179;
int S_MIN = 150;
int S_MAX = 255;
int V_MIN = 60;
int V_MAX = 255;

int CAPTURE_WIDTH = 1280;
int CAPTURE_HEIGHT =  720;
int MAX_OBJECT_AREA = CAPTURE_WIDTH*CAPTURE_HEIGHT/1.5;


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
        int numObjects = static_cast<int>(hierarchy.size());
        if(numObjects<MAX_NUM_OBJECTS) {
            for(int index=0; index >= 0; index = hierarchy[index][0]) {
                Moments moment = moments((cv::Mat)contours[index]);
                double area = moment.m00;
                if(area > MIN_OBJECT_AREA && area < MAX_OBJECT_AREA && area>refArea) {
                    x = static_cast<int>(moment.m10/area);
                    y = static_cast<int>(moment.m01/area);
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







StartTracking::StartTracking(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::StartTracking)
{
    ui->setupUi(this);

    connect(ui->btnExecuter,&QAbstractButton::click,this,&StartTracking::onBrowseFileIn);

    ui->cbShowHSV->setChecked(true);
    ui->cvShowTH->setChecked(true);
    ui->cvReduceNoise->setChecked(true);

    showHSV = true;
    showTS = true;
    filtreNoise = true;

}


void StartTracking::onAccept() {
    // Valide que les parametres entrée sont bon et demarre le tracking

}

void StartTracking::onReject() {
    this->close();
}

void StartTracking::onBrowseFileIn() {

}

void StartTracking::onBrowseFileOut() {

}

void StartTracking::onCheckBoxChecked() {

}

void StartTracking::onRadioBoxChecked() {

}


void StartTracking::start() {
    // Numero de la camera utilisé
     int device = 0;

     bool morphOps = true;

     cv::VideoCapture cap;
     std::string from;
     if(mode == FILE_SOURCE) {
        cv::VideoCapture c(inFile.toStdString());
        cap = c;
        from = "fichier "+inFile.toStdString();
        CAPTURE_HEIGHT = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
        CAPTURE_WIDTH = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
     } else {
         cv::VideoCapture c(device);
         cap = c;
         from = "camera " + std::to_string(device);
         cap.set(CAP_PROP_FRAME_WIDTH,CAPTURE_WIDTH);
         cap.set(CAP_PROP_FRAME_HEIGHT,CAPTURE_HEIGHT);
     }
     if(!cap.isOpened()) {
         std::cerr << "Impossible d'ouvrir la capture depuis " << from << std::endl;
         return;
     }

     cv::namedWindow(AppName);

     createMyTrackBar();

     cv::Mat img, imgHsv, imgThresh, imgMorph;

     int lastX = -1;
     int lastY = -1;

     while(cap.read(img)) {
         if(filtreNoise) { // Si on veut applique un filtre median pour réduire le bruit sur l'image
            // Fait notre traitement sur l'image
            cv::medianBlur(img,img,5);
         }
         // change l'image en hsv
         cvtColor(img,imgHsv,COLOR_BGR2HSV);

         // filtre l'image hsv selon nos valeurs de gaps
         inRange(imgHsv,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),imgThresh);

         // performe une operation morphologique pour eliminer du bruit des objects et des trous de l'avant-plan
         if(morphOps) {
             morphImg(imgThresh);
         }

         Moments oMoments = moments(imgThresh);

         double dM01 = oMoments.m01;
         double dM10 = oMoments.m10;
         double dArea = oMoments.m00;

         if(dArea > 10000) {
             // calcul la position de l'object
             int posX = dM10 / dArea;
             int posY = dM01 / dArea;

             if(lastX >= 0 && lastY > 0 && posX > 0 && posY > 0) {
                 line(img,Point(posX,posY),Point(lastX,lastY),Scalar(0,0,255),2);
             }

             if(posX <= 0 + 15 && posX < lastX) {
                 std::cout << "Item est sortit a gauche" << std::endl;
             }
             if(posX >= CAPTURE_WIDTH - 15 && posX > lastX && lastX != -1) {
                 std::cout << "Item est sortit a droite" << std::endl;
             }
             if(posY <= 0 + 15 && posY < lastY) {
                 std::cout << "Item est sortit en haut" << std::endl;
             }
             if(posY >= CAPTURE_HEIGHT - 15 && posY > lastY && lastY != -1) {
                 std::cout << "Item est sortit en bas" << std::endl;
             }
             lastX = posX;
             lastY = posY;

             std::cout << "X :" << posX << " Y :" << posY << std::endl;
         }

        //if(trackObject) {
        //     trackFilteredObject(x,y,imgThresh,img);
        // }

         if(showTS)
            imshow(wThres,imgThresh);
         if(showHSV)
            imshow(wHsv,imgHsv);
         imshow(AppName,img);


         if(parseKeyPress(cv::waitKey(30))) {
             // Quitte la boucle
             break;
         }
     }
     cap.release();
     cvDestroyAllWindows();

}

StartTracking::~StartTracking()
{
    delete ui;
}

void StartTracking::on_btnExecuter_clicked()
{
    start();
}

void StartTracking::on_btnQuitter_clicked()
{
   this->close();
}

void StartTracking::on_btnParcourirIn_clicked()
{
    QString filePath = QFileDialog::getOpenFileName(this,"Selectionner une image a charger",QStandardPaths::locate(QStandardPaths::MoviesLocation,""),"AVI (*.avi);;All Files (*)");
    if(filePath.isEmpty())
        return;
    ui->txtSrcFile->setText(filePath);
    inFile = filePath;
    ui->rbFile->setChecked(true);
    mode = FILE_SOURCE;
}
