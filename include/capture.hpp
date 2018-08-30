#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cuda.hpp>

#include <iostream>
#include <string>


using namespace std;
using namespace cv;


const string Extensions[] = {".jpg"};


void SharpenImage(const Mat& image, Mat& result);

void lolol() {
    std::cout << "there is " << cuda::getCudaEnabledDeviceCount() << " cuda device on this computer" << std::endl;
}

struct Resolution {
    int Width;
    int Heigth;
};

// ImageManipulator est une classe pour manipuler les pixels
// d'une image
class ImageWrapper {

    protected:
	    cv::Mat image;
	    std::string fileName;
        std::string errorBuffer;

        cv::Vec3b(*manipulator)(cv::Vec3b);

	public:
        ImageWrapper() { }
		ImageWrapper(std::string fileName) {
            this->fileName = fileName;
        }
        ImageWrapper(cv::Mat image) {
            this->image = image;
        }
        // Ouvre l'imagine et s'assure qu'elle existe
        bool Open() {
            if(this->fileName != "") {
                this->image = cv::imread(fileName);
            }
            if(this->image.empty()) {
                return false;
            }
            return true;
        }

        bool GetImageFromCamera(string fileName, int device,Resolution resolution) {
            cv::VideoCapture cap(device);
            if(!cap.isOpened()) {
		        return false;
            }
            cap.set(CAP_PROP_FRAME_HEIGHT,resolution.Heigth);
            cap.set(CAP_PROP_FRAME_WIDTH,resolution.Width);
            string windowName = "Capturing " + fileName;

        	cv::namedWindow(windowName);
        	cv::Mat img;

        	while(1) {
        		if(!cap.read(img)) {
        			std::cerr << "Error capturing " << std::endl;
        			return false;
        		}
                imshow(windowName,img);
        		if(cv::waitKey(0)) {
        			// set l'image comment etant elle saver
                    OnNewImage(img);
    			    break;
		        }
	       }
           return true;
        }

        void ShowAndWait() {
        	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
           	cv::imshow("Displaying "+this->fileName,image);
        	cv::waitKey(0);
        }

        bool Save(std::string outputName) {
            return cv::imwrite(outputName,this->image);
        }
    	// SetManipulatePixel 
		void SetManipulatePixel(cv::Vec3b(*manipulate)(cv::Vec3b)) {
            this->manipulator = manipulate;
        }
        

        void SharpenImageMask() {
            Mat m;
            SharpenImage(this->image,m);
            OnNewImage(m);
        }

        void SharpenImageFilter2D() {
            Mat m;
            Mat kernel = (Mat_<char>(3,3) << 0, -1, 0,
                                             -1, 5, -1,
                                              0, -1, 0);
            cv::filter2D(this->image,m,this->image.depth(),kernel);
            OnNewImage(m);
        }

        // TransformPixel ...
		void TransformPixel() {
            if(this->manipulator != NULL) {

            }
            for(int y = 0; y < this->image.cols;++y) 
		        for( int x = 0; x < this->image.cols;++x)
		        {
			        // get le pixel
			        Vec3b color = this->image.at<Vec3b>(Point(x,y));
			        // set le pixel avec le resultat de la fonction passé
			        this->image.at<Vec3b>(Point(x,y)) = this->manipulator(color);
		        }
        }

    protected:
        virtual void OnNewImage(cv::Mat img) {
            this->image = img;
        }
            
};


class ImagesWrapper : public ImageWrapper {
    int currentIndex;
    vector<cv::Mat> images;

    public:
        ImagesWrapper() : ImageWrapper() {
            // index commence a -1 pour dire qu'il n'y a pas de show a show
            currentIndex = -1;
        }

        bool PreviousImage() {
            if (this->currentIndex <= 0)
                return false;
            currentIndex--;
            this->images = images[currentIndex];
            return true;
        }

        bool NextImage() {
            if (this->currentIndex+1 >= this->images.size())
                return false;
            currentIndex++;
            this->images = images[currentIndex];
            return true;
        }
    protected:
        
        void OnNewImage(cv::Mat img) {
            images.push_back(this->image);
            this->currentIndex++;
            this->image = img;
        }
};


class ImageModifier : public ImagesWrapper {
    
    public:
        ImageModifier() {
            
        }
};


void SharpenImage(const Mat& image, Mat& result) {
    // throw une exception si l'image n'est pas de type uchar
    CV_Assert(image.depth() == CV_8U);

    // Crée une nouveau mat image de la grosseur et du type de l'image passé en argument
    result.create(image.size(), image.type());
    const int nbrCh = image.channels();

    for(int j = 1; j < image.rows -1;j++) {
        const uchar* previous = image.ptr<uchar>(j-1);
        const uchar* current  = image.ptr<uchar>(j);
        const uchar* next     = image.ptr<uchar>(j+1);

        uchar* output = result.ptr<uchar>(j);

        for(int i=nbrCh; i<nbrCh *(image.cols - 1); i++) {
            *output++ = cv::saturate_cast<uchar>(5* current[i]-current[i-nbrCh] - current[i+nbrCh] - previous[i] - next[i]);
        }

        result.row(0).setTo(Scalar(0));
        result.row(result.rows-1).setTo(Scalar(0));
        result.col(0).setTo(Scalar(0));
        result.col(result.cols-1).setTo(Scalar(0));
    }

}

