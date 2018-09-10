#include "cvheaders.hpp"



class PixelVideoModifier {
    private:
        int device = 0;
        std::string recordLocation;

        void iterateImage(cv::Mat& image) {
             for(int y = 0; y < image.rows;++y) 
	            for( int x = 0; x < image.cols;++x)
	            {
		            // get le pixel
					cv:Vec3b* vec = image.ptr<Vec3b>(x,y);
		            // set le pixel avec le resultat de la fonction passé
	                Modifier(vec);
	            }
        }

	protected:
		// Modifier est la fonction qui est appeler pour modifier mon pixel
		virtual void Modifier(cv::Vec3b *v) { }
		// HandleKey est la fonction qui parse les key entrer par l'usager
		virtual bool HandleKey(int k) {
			if(cv::waitKey(0)) {
				return true;
			}
			return false;
		}

    public:
		PixelVideoModifier() {}

        bool Start() {
            cv::VideoCapture cap(device);
			
            if(!cap.isOpened()) {
		        return false;
            }
            cap.set(CAP_PROP_FRAME_HEIGHT,1280);
            cap.set(CAP_PROP_FRAME_WIDTH,720);
            string windowName = "Pixel Video Modifier";

        	cv::namedWindow(windowName);
        	cv::Mat img;

        	while(1) {
        		if(!cap.read(img)) {
        			std::cerr << "Error capturing " << std::endl;
        			return false;
        		}
				// change l'image capturer de colorspace pour HSV
				//cv::cvtColor(img,CV_BGR2HSV);
				// applique la modification sur l'image
				iterateImage(img);
				// l'affiche
                imshow(windowName,img);
        		int k = cv::waitKey(0);
				if(HandleKey(k)) {
					break;
				}
	       }
           return true;
        }
    
};

