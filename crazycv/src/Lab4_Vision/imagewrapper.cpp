#include "imagewrapper.h"

#include "cvheaders.h"
#include <vector>


// Fonction de conversion de mon backend


float max(float a,float b,float c) {
    if( a > b && a > c) return a;
    if (b > a && b > c) return b;
    return c;
}

float min(float a,float b,float c) {
    if (a < b && a < c) return a;
    if (b < a && b < c) return b;
    return c;
}

void rgb_to_gs(cv::Vec3b* a,uchar* b,int size) {
    for(int i(0);i<size;i++) {
        float f = a[i][0] * 0.3f + a[i][1] * 0.59f + a[i][2] * 0.11f;
        b[i] = static_cast<uchar>(f);
    }
}

void rgb_to_hsv(cv::Vec3b* ArrayA,cv::Vec3f* ArrayR,int size) {
    for(int index=0;index<size;index++) {
        float Bs = ArrayA[index][0]/255.0f;
        float Gs = ArrayA[index][1]/255.0f;
        float Rs = ArrayA[index][2]/255.0f;

        float CMax = max(Bs,Gs,Rs);
        float CMin = min(Bs,Gs,Rs);

        float Delta = CMax - CMin;

        float h;
        if(Delta == 0) {
            h = 0;
        } else if(CMax == Rs) {
            h = 60 * (int((Gs-Bs)/Delta)%6);
        } else if (CMax == Gs) {
            h = 60 * (((Bs-Rs)/Delta)+2);
        } else { // Bs
            h = 60 * (((Rs-Gs)/Delta)+4);
        }

        float s;
        if (CMax == 0) {
            s = 0;
        } else {
            s = Delta/CMax;
        }

        ArrayR[index] = cv::Vec3f(h,s,CMax);
    }
}


void gs_to_bw(uchar* a,uchar* b,int size) {
    for(int i(0);i<size;i++) {
        if(a[i] > 125) {
            b[i] = 255;
        } else {
            b[i] = 0;
        }
    }
}



// IMPLEMENTATION IMAGEWRAPPER


void ImageWrapper::callImageChanged() {
    // Get la dernier image et si on peut next ou previous
    imageChanged(images.at(static_cast<unsigned int>(currentIndex)));
}

// Ajoute une nouveau image au bout de la queue
void ImageWrapper::appendImage(cv::Mat& m, AcquisionMode origin, ColorSpace color) {
    MyImage i;
    i.color = color;
    i.image = m;
    i.origin = origin;
    if(i.origin == FromFile) {
        i.isondisk = true;
    }
    images.push_back(i);
    // mets l'indice vers l'element ajouter
    currentIndex = static_cast<int>(images.size()-1);
    callImageChanged();
}

// Ajoute une nouveau image depuis une fichier
bool ImageWrapper::appendImageFromFile(std::string filePath) {
    if(filePath == "") return false;
    cv::Mat m = cv::imread(filePath);
    if(m.empty()) {
        return false;
    }
    cv::cvtColor(m,m,CV_BGR2RGB);
    // si l'image est empty on continue pas
    if(m.empty()) {
        // Affiche une message d'erreur        cv::cvtColor(*original,m,cv::COLOR_RGB2GRAY);
        return false;
    }
    appendImage(m,FromFile,RGB_CS);
    return true;
}

bool ImageWrapper::appendImageFromCamera(int device) {
    cv::VideoCapture cap(device);
    if(!cap.isOpened()) return false;

    cap.set(cv::CAP_PROP_FRAME_HEIGHT,600);
    cap.set(cv::CAP_PROP_FRAME_WIDTH,800);

    cv::Mat img;

    while(1) {
        if(!cap.read(img)) {
            return false;
        }
        if(cv::waitKey(0) == 'c') {
            cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
            appendImage(img,FromCamera,RGB_CS);
            return true;
        }
    }

}

// Obtient l'image a l'index courrant
MyImage ImageWrapper::getCurrentImage() {
    return images.at(static_cast<unsigned int>(currentIndex));
}

// Change pour l'image précédente si possible        
bool ImageWrapper::previousImage() {
    if(currentIndex <= 0) return false;
    currentIndex--;
    callImageChanged();
    return true;

}

// Change pour l'image suivante si possible
bool ImageWrapper::nextImage() {
    if(static_cast<unsigned int>(currentIndex) >= images.size()-1) return false;
    currentIndex++;
    callImageChanged();
    return true;
}

bool ImageWrapper::hasNext() {
    return (static_cast<unsigned int>(currentIndex) < images.size()-1);
}

bool ImageWrapper::hasPrevious() {
    return (currentIndex > 0);
}

// Retourne a l'image d'origine
bool ImageWrapper::returnFirstImage() {
    if (currentIndex < 0) return false;
    currentIndex = 0;
    callImageChanged();
    return true;
}

void ImageWrapper::reset() {
    currentIndex = -1;
    images.clear();
}


int ImageWrapper::getNbrImages() { return static_cast<int>(images.size()); }
int ImageWrapper::getCurrentIndex() { return currentIndex; }

// Enregistre l'image courrante dans un fichier
bool ImageWrapper::saveCurrentImage(const std::string filePath) {
    MyImage i = getCurrentImage();
    return cv::imwrite(filePath,i.image);
}


// IMPLÉMENTATION IMAGETRANSFORMER        
void ImageTransformer::toGS(cv::Mat& original,cv::Mat& m) {
    m.create(original.rows,original.cols,CV_8U);
    if(mode == OpencvBackend) {
        cv::cvtColor(original,m,cv::COLOR_RGB2GRAY);
    } if (mode == CustomBackend) {
        cv::Vec3b* dataA = original.ptr<cv::Vec3b>(0);
        uchar* dataR = m.ptr<uchar>(0);
        rgb_to_gs(dataA,dataR,original.rows*original.cols);
    }
}

void ImageTransformer::toHSV(cv::Mat& i,cv::Mat& m) {
    if(mode == OpencvBackend) {
        cv::cvtColor(i,m,cv::COLOR_RGB2HSV);
    } else { // Custom backend
        cv::Vec3b* dataA = i.ptr<cv::Vec3b>(0);
        cv::Vec3f* dataR = m.ptr<cv::Vec3f>(0);
        rgb_to_hsv(dataA,dataR,i.rows*i.cols);
    }
}
        
void ImageTransformer::transformPasseBas(cv::Mat&,cv::Mat&) {

}
void ImageTransformer::transformPasseHaut(cv::Mat&,cv::Mat&) {

}
void ImageTransformer::transformMedianne(cv::Mat&,cv::Mat&) {

}
void ImageTransformer::transformMoyenne(cv::Mat&,cv::Mat&) {

}
        
void ImageTransformer::setBackend(BackendMode mode) {
    this->mode = mode;
}

void ImageTransformer::setTransformMatriceSize(int s) {
    if(s == 3 || s == 5)
        matriceSize = s;

}

bool ImageTransformer::getColorHistogramme(cv::Mat& m,ColorHistogramme* h) {
    // Valide que l'image aye 3 channels et soit en rgb
    if(m.channels()!=3) return false;
    for(int i=0;i<257;++i) {
        h->Red[i] = 0;
        h->Green[i] = 0;
        h->Blue[i] = 0;
    }

    int size = m.cols * m.rows;
    cv::Vec3b* vec = m.ptr<cv::Vec3b>(0);

    for(int i=0;i<size;i++) {
        uchar r = vec[i][0];
        uchar g = vec[i][1];
        uchar b = vec[i][2];

        h->Red[r] = ++h->Red[r];
        h->Green[g] = ++h->Green[g];
        h->Blue[b] = ++h->Blue[b];
    }
    for(int i=0;i<257;i++) {
        printf("R %d = %d G %d = %d B %d = %d\n",i,h->Red[i],i,h->Green[i],i,h->Blue[i]);
    }
    return true;
}

