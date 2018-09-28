#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
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
    void SetImage(cv::Mat& m);
    Ui::MainWindow *ui;

private slots:
    void loadFileFromCamera();
    void loadFileFromFile();
    void previousImage();
    void nextImage();
    void saveImage();
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
