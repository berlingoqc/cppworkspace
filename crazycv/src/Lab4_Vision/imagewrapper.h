#ifndef IMAGEWRAPPER_H
#define IMAGEWRAPPER_H

#include "cvheaders.h"


// AcquisionMode correspond au différente origine possible d'une image
enum AcquisionMode { FromFile, FromCamera, FromTransformation };

struct ColorHistogramme {
    int Red[257];
    int Green[257];
    int Blue[257];
};

bool GetColorHistogramme(cv::Mat&,ColorHistogramme*);

class ImageWrapper
{

private:
    AcquisionMode acquisionMode;


    std::string filePath;

    cv::Mat currentImage;
    std::vector<cv::Mat> previousImage;

    int indexImage;

    std::string errorMessage;

private:
    void appendNewImage(cv::Mat mat);

public:
    ImageWrapper();
    ImageWrapper(std::string filePath);
    ImageWrapper(cv::Mat& mat);

    bool IsEmpty();
    bool HasNext();
    bool HasPrevious();

    bool SwitchPrevious();
    bool SwitchNext();
    bool ResetOriginal();

    bool OpenFile(std::string filePath);
    bool Save(std::string filePath);
    bool AquireFromCamera(int index);

    cv::Mat& GetCurrentImage();

};

#endif // IMAGEWRAPPER_H
