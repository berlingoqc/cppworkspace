#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>




MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->lblImage->setScaledContents(true);
    ui->lblImage->setSizePolicy( QSizePolicy::Ignored, QSizePolicy::Ignored );

    cv::Mat image = cv::imread("D://test.jpg");
    SetImage(image);
}

void MainWindow::SetImage(cv::Mat& m) {

    QImage qI(m.data,m.cols,m.rows, static_cast<int>(m.step), QImage::Format_RGB888);
    QPixmap qPm(QPixmap::fromImage((qI)));
    // Set la grosseur du label et de la fenetre la grosseur
    // de mon image

    ui->lblImage->setPixmap(qPm);
    return;
}


MainWindow::~MainWindow()
{
    delete ui;
}
