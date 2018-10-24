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


private slots:
    void loadFileFromCamera();
    void loadFileFromFile();
    void previousImage();
    void nextImage();
    void saveImage();
    void resetOriginalImage();
    void closeImage();
    void quitApp();

    void selectCustomBackend();
    void selectOpencvBackend();


    void toHSV();
    void toBW();
    void toGS();

    void showColorError(QString mustBe, QString is);
    void showHistogramme();
    void showDetailImage();

    void transformPasseBas();
    void transformPasseHaut();

    void transformMedianne();
    void transformMoyenne();

    void updateImage(MyImage img);
};

#endif // MAINWINDOW_H
