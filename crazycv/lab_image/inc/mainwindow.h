#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "imagelabel.h"
#include "imagewrapper.h"
#include "cvheaders.h"


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    void validPreviousNext();
    void updateStatusBar(const MyImage& i);

    ImageWrapper 		img;
    ImageTransformer 	transformer;

    QLabel* 			lblMyImageInfo;
    QLabel*				lblMatInfo;
    ImageLabel* 		lblImage;
    Ui::MainWindow*		ui;

private:
    void openColorHistogramme(const cv::Mat& m);

private slots:
    void loadFileFromCamera();
    void loadFileFromFile();
    void previousImage();
    void nextImage();
    void saveImage();
    void resetOriginalImage();
    void closeImage();
    void quitApp();



    void setDetectMilleFeuille();
    void detectMilleFeuille();
    void videoTracking();

    void selectCustomBackend();
    void selectOpencvBackend();


    void toHSV();
    void toBW();
    void toGS();

    void showError(QString message);
    void showHistogramme();
    void showDetailImage();

    void transformPasseBas();
    void transformPasseHaut();

    void transformMedianne();
    void transformMoyenne();

    void selectKernel3();
    void selectKernel5();

    bool validBeforeOperation();

    void updateImage(MyImage img);
};

#endif // MAINWINDOW_H
