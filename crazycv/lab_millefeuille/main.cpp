#include "confmillefeuille.h"
#include "cvheaders.h"

#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;

// La liste des mes images sources 
std::vector<millefeuille_image> images = {
    { "/home/wq/mille_feuille/mille_feuille_1.jpg", TOP_VIEW, OK_STATE },
    { "/home/wq/mille_feuille/mille_feuille_2.jpg", TOP_VIEW, TOO_SMALL_STATE},
    { "/home/wq/mille_feuille/side_no_top.jpg", SIDE_VIEW, NOT_TOP_STATE}, // hreshHold : H_MIN 6 H_MAX 25 S_MIN 123 S_MAX 255 V_MIN 32V_MAX255
    { "/home/wq/mille_feuille/side_no_top2.jpg", SIDE_VIEW, NOT_CREME_STATE},
    { "/home/wq/mille_feuille/side_ok.jpg", SIDE_VIEW, OK_STATE},
    { "/home/wq/mille_feuille/op.jog", TOP_VIEW, OK_STATE}
};


static const char * wThres = "Image Thresholded";
static const char * wCanny = "Canny edge";


// initialise les maximums et minimum pour hsv
int H_MIN = 0;
int H_MAX = 255;
int S_MIN = 140;
int S_MAX = 255;
int V_MIN = 0;
int V_MAX = 255;

int T_MIN = 100;
int T_MAX = 200;

RNG rng(12345);

void on_trackbar(int t, void*)  {

}

// appliquer une errotion et une dilatation
void morphImg(cv::Mat& threshold) {
    // Crée les éléments pour appliquer une matrice de dilatation et d'erosion
    cv::Mat erodeElement = getStructuringElement(MORPH_RECT,Size(3,3));
    cv::Mat dilateElement = getStructuringElement(MORPH_RECT,Size(3,3));

    cv::erode(threshold,threshold,erodeElement);
    cv::erode(threshold,threshold,erodeElement);

    cv::dilate(threshold,threshold,dilateElement);
    cv::dilate(threshold,threshold,dilateElement);
}

void createMyTrackBar() {
    cv::namedWindow(wThres);
    cv::namedWindow(wCanny);

    createTrackbar( "H_MIN", wThres, &H_MIN, H_MAX, on_trackbar );
    createTrackbar( "H_MAX", wThres, &H_MAX, H_MAX, on_trackbar );
    createTrackbar( "S_MIN", wThres, &S_MIN, S_MAX, on_trackbar );
    createTrackbar( "S_MAX", wThres, &S_MAX, S_MAX, on_trackbar );
    createTrackbar( "V_MIN", wThres, &V_MIN, V_MAX, on_trackbar );
    createTrackbar( "V_MAX", wThres, &V_MAX, V_MAX, on_trackbar );

    createTrackbar ( "T_MIN",wCanny,&T_MIN,T_MAX,on_trackbar);
    createTrackbar ( "T_MAX",wCanny,&T_MAX,T_MAX,on_trackbar);
}

void findThreshOld(const millefeuille_image& mfl) {
    std::stringstream ss;
    ss << "Image Path " << mfl.filename << " Vue " << mfl.view << " Etat " << mfl.view;
    createMyTrackBar();

    Mat t = cv::imread(mfl.filename);
    if(t.empty()) {
        std::cerr << "Ne trouve pas le fichier " << mfl.filename << std::endl;
        return;
    }

    // Convertit t en hsv
    cvtColor(t,t,COLOR_BGR2HSV);

    Size s(800 , 600 );

    cv::resize(t,t,s);



    vector<vector<Point>>   contours;
    vector<Vec4i>           hierarchy;   

    Mat thresh;
    Mat bin;

    Mat kp;

    while(true) {

        // Convertit l'image en threshold
        inRange(t,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),thresh);
        // Applique une blur pour reduire le bruit
        //morphImg(thresh);
        blur(thresh,thresh,Size(5,5));
        cv::threshold(thresh,thresh,128,255,CV_THRESH_BINARY);
        // Affiche les informations sur l'image
        putText(thresh, ss.str().c_str(), cvPoint(30,30), FONT_HERSHEY_PLAIN, 1.0, Scalar(0,255,0), 1);

        cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(40, 40));
        cv::morphologyEx( thresh, thresh, cv::MORPH_CLOSE, structuringElement );

        Canny(thresh,kp,T_MIN,T_MIN * 2,3);
        findContours( kp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE,Point(0,0));

        // Connect les composantes avec 8 way connectivité
        int v = connectedComponents(kp,kp,8,4);
        printf("There is %d label \n",v);
        /// Get the moments
        vector<Moments> mu(contours.size() );
        for( int i = 0; i < contours.size(); i++ )
        { mu[i] = moments( contours[i], false ); }
 
        ///  Get the mass centers:
        vector<Point2f> mc( contours.size() );
        for( int i = 0; i < contours.size(); i++ )
        { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }
 
        /// Draw contours
        Mat drawing = Mat::zeros( kp.size(), CV_8UC3 );
        for( int i = 0; i< contours.size(); i++ )
        {
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
            circle( drawing, mc[i], 4, color, -1, 8, 0 );
        }
 
        /// Show in a window
        namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
        imshow( "Contours", drawing );
 
        /// Calculate the area with the moments 00 and compare with the result of the OpenCV function
        imshow("f",t);
        imshow(wThres,thresh);
        imshow(wCanny, kp);
        int k = cv::waitKey(0);
        if(k == 'b') {
            printf("\t Info: Area and Contour Length \n");
            for( int i = 0; i< contours.size(); i++ )
                printf(" * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Lenght: %.2f \n", i , mu[i].m00, contourArea(contours[i]), arcLength(contours[i], true));
        } else if(k != 'c') {
            std::cout << "ThreshHold : H_MIN " << H_MIN << " H_MAX " << H_MAX << " S_MIN "<< S_MIN << " S_MAX " << S_MAX << " V_MIN " << V_MIN << "V_MAX" << V_MAX << std::endl;
            return;
        }
    }
}


int main(int argc, char const *argv[])
{
    for(auto i : images) {
        findThreshOld(i);
    }
    return 0;
}
