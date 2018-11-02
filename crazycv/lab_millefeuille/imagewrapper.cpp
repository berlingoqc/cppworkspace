#include "imagewrapper.h"
#include "cvheaders.h"
#include <vector>


// Fonction de conversion de mon backend

const char kernelPassHaut3x3[3][3] { {0, -1 , 0}, {-1, 5, -1}, {0, -1, 0} };
const char kernelPasseBas3x3[3][3] { {1, 1, 1},{ 1, 4, 1},{1, 1, 1}};
const char kerneldefault3x3[3][3] { {1, 1, 1}, {1, 1, 1},{ 1, 1, 1}};
const char kerneldefault5x5[5][5] {{1,1,1,1,1}, {1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}};


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
        float Bs = ArrayA[index][2]/255.0f;
        float Gs = ArrayA[index][1]/255.0f;
        float Rs = ArrayA[index][0]/255.0f;

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

void apply_filtre_median(cv::Mat& MyImage,cv::Mat& out,int size) {
    uint nbrElement = static_cast<unsigned int>(size * size);
    int offset = 1;
    if(size == 5)
        offset = 2;

    std::vector<uchar> testValue(nbrElement+1);
    for(int j=offset;j<MyImage.rows-offset; ++j) {
        const uchar* previous2;
        const uchar* previous = MyImage.ptr<uchar>(j-1);
        const uchar* current = MyImage.ptr<uchar>(j);
        const uchar* next = MyImage.ptr<uchar>(j+1);
        const uchar* next2;

        if(offset == 2) {
            previous2 = MyImage.ptr<uchar>(j-2);
            next2 = MyImage.ptr<uchar>(j+2);
        }

        uchar* output = out.ptr<uchar>(j);
        for(int i=offset; i < MyImage.cols -offset;++i) {
            // va chercher les valeurs de mon kernel de 3x3 et si besoin apres va chercher les autres points
            for(int z=i-offset;z<=i+offset;z++) {
                testValue.push_back(previous[i]);
                testValue.push_back(current[i]);
                testValue.push_back(next[i]);
                if(offset == 2) {
                    testValue.push_back(previous2[i]);
                    testValue.push_back(next2[i]);
                }
            }
            // Ajout une valeur aberante
            testValue.push_back(255);
            // Sort la list
            std::sort(testValue.begin(),testValue.end());
            // assigne la valeur central comme output
            output[i] = testValue.at((nbrElement+1)/2);
            testValue.clear();
        }
    }
}

// Passer le mode 0 pour effectuer une moyenne, size correspond a la grandeur de mon kernel
template<size_t N>
void apply_filtre_moyenne(cv::Mat& MyImage,cv::Mat& out, int size,int mode,const char (&kernel)[N][N]) {
    uint nbrElement = 1;
    if(mode == 0) {
        nbrElement = static_cast<unsigned int>(size * size);
    }
    int offset = 1;
    if(size == 5)
        offset = 2;

    for(int j=offset;j<MyImage.rows-offset; ++j) {
        const uchar* previous2;
        const uchar* previous = MyImage.ptr<uchar>(j-1);
        const uchar* current = MyImage.ptr<uchar>(j);
        const uchar* next = MyImage.ptr<uchar>(j+1);
        const uchar* next2;

        uchar klv = 0;
        if(offset == 2) {
            previous2 = MyImage.ptr<uchar>(j-2);
            next2 = MyImage.ptr<uchar>(j+2);
            klv = 1;
        }

        uchar* output = out.ptr<uchar>(j);

        uchar v;
        uchar k = 0;



        for(int i=offset; i < MyImage.cols -offset;++i) {
            // va chercher les valeurs de mon kernel de 3x3 et si besoin apres va chercher les autres points
            v = 0;
            k = 0;

            for(int z=i-offset;z<=i+offset;z++) {
                v += previous[i] * kernel[klv][k];
                v += current[i] * kernel[klv+1][k];
                v += next[i] * kernel[klv+2][k];
                if(offset==2) {
                    v += previous2[klv-1];
                    v += next2[klv+3];
                }
                k += 1;
            }
            output[i] = cv::saturate_cast<uchar>( v / nbrElement );
        }
    }
}

// IMPLEMENTATION IMAGEWRAPPER


void ImageWrapper::callImageChanged() {
    // Get la dernier image et si on peut next ou previous
    #ifdef WITH_QT_SLOTS
        imageChanged(images.at(static_cast<unsigned int>(currentIndex)));
    #endif
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
    } else if (mode == CustomBackend) {
        cv::Vec3b* dataA = original.ptr<cv::Vec3b>(0);
        uchar* dataR = m.ptr<uchar>(0);
        rgb_to_gs(dataA,dataR,original.rows*original.cols);
    }
}

void ImageTransformer::toBW(cv::Mat& i,cv::Mat& m) {
    m.create(i.rows,i.cols,CV_8U);
    if(mode == OpencvBackend) {
        cv::threshold(i,m,100,255,cv::THRESH_BINARY);
    } else if( mode == CustomBackend) {
        uchar* dataA = i.ptr<uchar>(0);
        uchar* dataR = m.ptr<uchar>(0);
        gs_to_bw(dataA,dataR,i.rows*i.cols);
    }
}

void ImageTransformer::toHSV(cv::Mat& i,cv::Mat& m) {
    if(mode == OpencvBackend) {
        cv::cvtColor(i,m,cv::COLOR_RGB2HSV);
    } else { // Custom backend
        m.create(i.rows,i.cols,CV_32FC3);
        cv::Vec3b* dataA = i.ptr<cv::Vec3b>(0);
        cv::Vec3f* dataR = m.ptr<cv::Vec3f>(0);
        rgb_to_hsv(dataA,dataR,i.rows*i.cols);
    }
}
        
void ImageTransformer::transformPasseBas(cv::Mat& i,cv::Mat& o) {
    o = i.clone();
    if(mode == OpencvBackend) {
        cv::blur(i,o,cv::Size(matriceSize,matriceSize),cv::Point(-1,-1));
    } else {
        if(matriceSize == 3) {
            apply_filtre_moyenne<3>(i,o,matriceSize,0,kernelPasseBas3x3);
        } else {
            apply_filtre_moyenne<5>(i,o,matriceSize,0,kerneldefault5x5);
        }
    }
}
void ImageTransformer::transformPasseHaut(cv::Mat& i,cv::Mat& o) {
    o = i.clone();
    if(mode == OpencvBackend) {
        cv::blur(i,o,cv::Size(matriceSize,matriceSize),cv::Point(-1,-1));
    } else {
        if(matriceSize == 3) {
            apply_filtre_moyenne<3>(i,o,matriceSize,0,kernelPassHaut3x3);
        } else {
            apply_filtre_moyenne<5>(i,o,matriceSize,0,kerneldefault5x5);
        }
    }
}
void ImageTransformer::transformMedianne(cv::Mat& i,cv::Mat& o) {
    o = i.clone();
    if(mode == OpencvBackend) {
        cv::medianBlur(i,o,matriceSize);
    } else {
        apply_filtre_median(i,o,matriceSize);
    }

}
void ImageTransformer::transformMoyenne(cv::Mat& i,cv::Mat& o) {
    o = i.clone();
    if(mode == OpencvBackend) {
        cv::blur(i,o,cv::Size(matriceSize,matriceSize),cv::Point(-1,-1));
    } else {
        if(matriceSize == 3) {
            apply_filtre_moyenne<3>(i,o,matriceSize,0,kerneldefault3x3);
        } else {
            apply_filtre_moyenne<5>(i,o,matriceSize,0,kerneldefault5x5);
        }
    }
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

