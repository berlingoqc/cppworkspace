#include "../../../include/capture.hpp"
#include "../../../include/file.hpp"
#include <tclap/CmdLine.h>


void manipulateRedGreenBlue(cv::Mat& image) {
 for(int y = 0; y < image.rows;++y) 
	for( int x = 0; x < image.cols;++x)
	{
		// get le pixel
		Vec3b vec = image.at<Vec3b>(Point(x,y));
		// set le pixel avec le resultat de la fonction passé
        // Fait la manipluation sur le pixel
        int r = vec[2];
        int g = vec[1];
        int b = vec[0];
        //printf("X : %d Y : %d R : %d G: %d B: %d\n",x,y,r,v,b);
	    // Detection du rouge
	    if (r > 140 && r <= 255 && g < 100 & b < 100) {
           	vec = Vec3b(0,0,0);
    	} else if ( r > 240 && g > 240 && b < 130) {
			vec = Vec3b(0,0,0);
		} else if ( g > 140 && r < 100 & b < 100 ) {
			vec = Vec3b(0,0,0);
		}
	    image.at<Vec3b>(Point(x,y)) = vec;
	}
}

int main(int argv,char ** argc)
{
	std::string fileName;
	std::string outputFileName;

	int width = 1280;
	int height = 720;
	  cv::VideoCapture cap(0);
            if(!cap.isOpened()) {
		        return 1;
            }
            cap.set(CAP_PROP_FRAME_HEIGHT,height);
            cap.set(CAP_PROP_FRAME_WIDTH,width);
            string windowName = "Capturing " + fileName;

        	cv::namedWindow(windowName);

        	while(true) {

        		cv::Mat img;
				cap >> img;
				manipulateRedGreenBlue(img);
                imshow(windowName,img);
        		if(cv::waitKey(30) >= 0) {
        			// set l'image comment etant elle save
    			    break;
		        }
			}
}