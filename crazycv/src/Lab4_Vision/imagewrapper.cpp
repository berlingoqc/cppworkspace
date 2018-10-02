#include "imagewrapper.h"

#include "cvheaders.h"
#include <vector>

bool GetColorHistogramme(cv::Mat& m, ColorHistogramme* h) {
    // Valide que l'image aye 3 channels et soit en rgb
    if(m.channels()!=3) return false;

    int size = m.cols * m.rows;
    cv::Vec3b* vec = m.ptr<cv::Vec3b>(0);

    for(int i=0;i<size;i++) {
        uchar r = vec[i][0];
        uchar g = vec[i][1];
        uchar b = vec[i][2];

        h->Red[r] = h->Red[r]++;
        h->Green[g] = h->Green[g]++;
        h->Blue[b] = h->Blue[b]++;
    }
    return true;
}


ImageWrapper::ImageWrapper()
{

}

ImageWrapper::ImageWrapper(std::string filePath) {
    this->filePath = filePath;
    cv::Mat m = cv::imread(filePath);
    cv::cvtColor(m,m,CV_BGR2RGB);
}

ImageWrapper::ImageWrapper(cv::Mat& mat) {
    currentImage = mat;
}

bool ImageWrapper::IsEmpty() {
    return currentImage.empty() && previousImage.size();
}

bool ImageWrapper::HasNext() {
    return false;
}

bool ImageWrapper::HasPrevious() {
    return false;
}

bool ImageWrapper::SwitchPrevious() {
    return false;
}

bool ImageWrapper::SwitchNext() {
    return false;
}

bool ImageWrapper::ResetOriginal() {
    // get la premiere image du vecteur et la met comme image courrante
    if(previousImage.size()==0) return false;
    currentImage = previousImage.at(0);
    previousImage.clear();
}


bool ImageWrapper::OpenFile(std::string filePath) {
    this->filePath = filePath;
    if(filePath == "") return false;
    cv::Mat m = cv::imread(filePath);
    cv::cvtColor(m,m,CV_BGR2RGB);
    // si l'image est empty on continue pas
    if(m.empty()) return false;

    appendNewImage(m);
}

bool ImageWrapper::Save(std::string filePath) {
    if(currentImage.empty()) return false;
    return cv::imwrite(filePath,currentImage);
}

bool ImageWrapper::AquireFromCamera(int index) {

    cv::VideoCapture cap(index);

    if(!cap.isOpened()) {
        errorMessage = "Video capture device can't be open";
        return false;
    }

    cv::namedWindow("Capture d'une image, appuyer sur enter");
    cv::Mat img;

    while(true) {
        if(!cap.read(img)) {
            errorMessage = "Erreur durant la capture";
            return false;
        }

        if(cv::waitKey(30) > 0) {
            // Ajoute cette image comme nouvelle image
            appendNewImage(img);
            return true;
        }
    }

}

cv::Mat& ImageWrapper::GetCurrentImage() {
    return currentImage;
}

void ImageWrapper::appendNewImage(cv::Mat mat) {
    // si l'image mat courrante n'est pas vide on l'ajoute a la liste
    if(!currentImage.empty()) {
        previousImage.push_back(mat);
    }
    currentImage = mat;
}
