#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "imagelabel.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>




MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ImageLabel *il = new ImageLabel();
    ui->setupUi(this);
    setCentralWidget(il);

    cv::Mat image = cv::imread("D://test.jpg");
    cv::cvtColor(image,image,cv::COLOR_BGR2RGB);
    il->SetMat(image);
    //SetImage(image);
}

void MainWindow::SetImage(cv::Mat& m) {

    QImage qI(m.data,m.cols,m.rows, static_cast<int>(m.step), QImage::Format_RGB888);
    QPixmap qPm(QPixmap::fromImage((qI)));
    // Set la grosseur du label et de la fenetre la grosseur
    // de mon image

    //ui->lblImage->setPixmap(qPm);
    return;
}


void MainWindow::loadFileFromCamera() {}
void MainWindow::loadFileFromFile() {}
void MainWindow::previousImage() {}
void MainWindow::nextImage() {}
void MainWindow::saveImage() {}
void MainWindow::closeImage() {}
void MainWindow::quitApp() {}

void MainWindow::selectCustomBackend() {}
void MainWindow::selectOpencvBackend() {}

void MainWindow::toHSV() {}
void MainWindow::toBW() {}
void MainWindow::toGS() {}

void MainWindow::showHistogramme() {}
void MainWindow::showDetailImage() {}

void MainWindow::transformPasseBas() {}
void MainWindow::transformPasseHaut() {}

void MainWindow::transformMedianne()  {}
void MainWindow::transformMoyenne() {}


MainWindow::~MainWindow()
{
    delete ui;
}
