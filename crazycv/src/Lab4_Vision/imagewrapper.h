#ifndef IMAGEWRAPPER_H
#define IMAGEWRAPPER_H

#include "cvheaders.h"

#include <QObject>

enum BackendMode {
    CustomBackend, OpencvBackend
};

inline const char * ToString(BackendMode m) {
    switch (m) {
        case CustomBackend: return "Custom";
        case OpencvBackend: return "Opencv";
        default: return "";
    }
}

enum ColorSpace {
    RGB_CS, GS_CS, BW_CS, HSV_CS
};

inline const char * ToString(ColorSpace m) {
    switch (m) {
        case RGB_CS: 	return "RGB";
        case HSV_CS: 	return "HSV";
        case GS_CS:		return "GS";
        case BW_CS:		return "BW";
        default: return "";
    }
}

// AcquisionMode correspond au différente origine possible d'une image
enum AcquisionMode { FromFile, FromCamera, FromTransformation };

inline const char * ToString(AcquisionMode m) {
    switch (m) {
        case FromFile: 				return "File";
        case FromCamera: 			return "Camera";
        case FromTransformation: 	return "Transformation";
        default: return "";
    }
}

struct ColorHistogramme {
    int Red[257];
    int Green[257];
    int Blue[257];
};



struct MyImage {
    cv::Mat 		image;
    ColorSpace 		color;
    AcquisionMode 	origin;
    bool 			isondisk;
    const char*		filePath;
};

Q_DECLARE_METATYPE(MyImage);


class ImageWrapper : public QObject {

    Q_OBJECT
    private:

        int imageColorMode = 0;

        // List de mes images dans ma structure qui l'englobe
        std::vector<MyImage> images;
        int currentIndex = -1;

    private:
        // Wrapper autour de l'appel de la fonction slot imageChanged
        void callImageChanged();

    public:
        // Ajoute une nouveau image au bout de la queue
        void appendImage(cv::Mat& m, AcquisionMode origin, ColorSpace color);
        // Ajoute une nouveau image depuis une fichier
        bool appendImageFromFile(std::string);
        // Ajout une image depuis une camera
        bool appendImageFromCamera(int device);

        // Obtient l'image a l'index courrant
        MyImage getCurrentImage();

        // Change pour l'image précédente si possible        
        bool previousImage();
        // Change pour l'image suivante si possible
        bool nextImage();


        bool hasNext();
        bool hasPrevious();

        // Retourne a l'image d'origin
        bool returnFirstImage();
        // Efface toute les images de la mémoire
        void reset();


        int getNbrImages();
        // Get l'index courant de la photo
        int getCurrentIndex();

        // Enregistre l'image courrante dans un fichier
        bool saveCurrentImage(std::string filePath);

    signals:
        // Signale quand l'image est updater
        void imageChanged(MyImage img);
};

class ImageTransformer {

    private:
        // Mode de backend utiliser pour faire les traitements
        BackendMode mode = OpencvBackend;
        int matriceSize = 3;

    public:
        bool getColorHistogramme(cv::Mat&,ColorHistogramme*);
        
        void toGS(cv::Mat&,cv::Mat&);
        void toHSV(cv::Mat&,cv::Mat&);
        
        void transformPasseBas(cv::Mat&,cv::Mat&);
        void transformPasseHaut(cv::Mat&,cv::Mat&);
        void transformMedianne(cv::Mat&,cv::Mat&);
        void transformMoyenne(cv::Mat&,cv::Mat&);
        
        void setBackend(BackendMode mode);
        void setTransformMatriceSize(int s);


};

#endif // IMAGEWRAPPER_H
