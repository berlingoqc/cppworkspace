#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "imagelabel.h"
#include "cvheaders.h"

enum BackendMode {
    CustomBackend, OpencvBackend
};

enum ColorSpace {
    RGB_CS, HSV_CS, GS_CS, BW_CS
};

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
    void SetImage();
    void ValidPreviousNext();


    ImageLabel* lblImage;
    Ui::MainWindow *ui;

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

    void showHistogramme();
    void showDetailImage();

    void transformPasseBas();
    void transformPasseHaut();

    void transformMedianne();
    void transformMoyenne();
};

#endif // MAINWINDOW_H
