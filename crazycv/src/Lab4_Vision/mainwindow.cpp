#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "imagelabel.h"
#include "cvheaders.h"

#include <QFileDialog>
using namespace cv;

// Mode de backend utiliser pour faire les traitements
BackendMode mode = OpencvBackend;
int imageColorMode = 0;

// List des couleurs des images du meme index de l'autre liste
std::vector<ColorSpace> colors;
std::vector<cv::Mat> images;
int currentIndex = -1;


void rgb_to_gs(cv::Vec3b* a,uchar* b,int size) {
    for(int i(0);i<size;i++) {
        float f = a[i][0] * 0.3 + a[i][1] * 0.59 + a[i][2] * 0.11;
        b[i] = static_cast<uchar>(f);
    }
}


void gs_to_bw(uchar* a,uchar* b,int size) {
    for(int i(0);i<size;i++) {
        if(a[i] > 125) {
            a[i] = 255;
        } else {
            a[i] = 0;
        }
    }
}




// appendImage ajoute une nouvelle image a la fin de la liste et ajuste l'index pour allez a cette photo
void appendImage(cv::Mat m) {
    images.push_back(m);
    currentIndex = images.size()-1;
}


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    lblImage = new ImageLabel();
    ui->setupUi(this);
    setCentralWidget(lblImage);

    ui->actionPrecedente->setEnabled(false);
    ui->actionSuivante->setEnabled(false);

    connect(ui->actionDepuis_fichier,&QAction::triggered, this, &MainWindow::loadFileFromFile);
    connect(ui->actionDepuis_cam_ra,&QAction::triggered,this,&MainWindow::loadFileFromCamera);
    connect(ui->actionPrecedente,&QAction::triggered,this,&MainWindow::previousImage);
    connect(ui->actionSuivante,&QAction::triggered,this,&MainWindow::nextImage);
    connect(ui->actionR_initialiser_Origine,&QAction::triggered,this,&MainWindow::resetOriginalImage);
    connect(ui->actionSauvegarder,&QAction::triggered,this,&MainWindow::saveImage);
    connect(ui->actionQuitter,&QAction::triggered,this,&MainWindow::quitApp);

    connect(ui->actionCustom,&QAction::triggered,this,&MainWindow::selectCustomBackend);
    connect(ui->actionOpenCV,&QAction::triggered,this,&MainWindow::selectOpencvBackend);

    connect(ui->actionVers_HSV,&QAction::triggered,this,&MainWindow::toHSV);
    connect(ui->actionVers_Grayscale,&QAction::triggered,this,&MainWindow::toGS);
    connect(ui->actionVers_Black_White,&QAction::triggered,this,&MainWindow::toBW);

    connect(ui->actionHistogramme_couleur,&QAction::triggered,this,&MainWindow::showHistogramme);
    connect(ui->actionD_taille_image,&QAction::triggered,this,&MainWindow::showDetailImage);

    connect(ui->actionPasse_bas,&QAction::triggered,this,&MainWindow::transformPasseBas);
    connect(ui->actionPasse_haut,&QAction::triggered,this,&MainWindow::transformPasseHaut);

    connect(ui->actionM_dianne,&QAction::triggered,this,&MainWindow::transformMedianne);
    connect(ui->actionMoyenne,&QAction::triggered,this,&MainWindow::transformMoyenne);


}

void MainWindow::ValidPreviousNext() {
    bool next = false; bool previous = false;
    if(currentIndex < images.size()-1) {
        next = true;
    }
    if (currentIndex > 0) {
        previous = true;
    }
    ui->actionSuivante->setEnabled(next);
    ui->actionPrecedente->setEnabled(previous);
}

void MainWindow::SetImage() {
    if(currentIndex < 0) return;
    cv::Mat m = images.at(currentIndex);
    lblImage->SetMat(m,imageColorMode);
    ValidPreviousNext();
}

void MainWindow::loadFileFromCamera() {

}

void MainWindow::loadFileFromFile() {

    QString filePath = QFileDialog::getOpenFileName(this,"Selectionner une image a charger","","JPG (*.jpg);;All Files (*)");
    if(filePath == "") return;
    cv::Mat m = cv::imread(filePath.toStdString());
    cv::cvtColor(m,m,CV_BGR2RGB);
    // si l'image est empty on continue pas
    if(m.empty()) {
        // Affiche une message d'erreur        cv::cvtColor(*original,m,cv::COLOR_RGB2GRAY);

        return;
    }
    appendImage(m);
    SetImage();
}

void MainWindow::previousImage() {
    if(currentIndex <= 0) return;
    currentIndex--;
    SetImage();
    ValidPreviousNext();
}

void MainWindow::nextImage() {
    if(currentIndex >= images.size()-1) return;
    currentIndex++;
    SetImage();
    ValidPreviousNext();
}

void MainWindow::saveImage() {}

void MainWindow::resetOriginalImage() {
    currentIndex = 0;
    SetImage();
}

void MainWindow::closeImage() {
    currentIndex = -1;
    images.clear();
    ValidPreviousNext();
    lblImage->clear();
}

void MainWindow::quitApp() {
    this->close();
}

void MainWindow::selectCustomBackend() {
    mode = CustomBackend;
}
void MainWindow::selectOpencvBackend() {
    mode = OpencvBackend;
}

void MainWindow::toHSV() {
    if(mode == OpencvBackend) {
        cv::Mat m = images.at(currentIndex).clone();
        cv::cvtColor(images.at(currentIndex),m,cv::COLOR_RGB2HSV);
        appendImage(m);
        SetImage();
    } else {

    }
}
void MainWindow::toBW() {

    if(mode == OpencvBackend) {
        cv::Mat m = images.at(currentIndex).clone();

    }
}
void MainWindow::toGS() {
    cv::Mat original = images.at(currentIndex);
    cv::Mat m(original.rows,original.cols,CV_8U);
    if(mode == OpencvBackend) {
        m = original.clone();
        cv::cvtColor(original,m,cv::COLOR_RGB2GRAY);
    } if (mode == CustomBackend) {
        cv::Vec3b* dataA = original.ptr<cv::Vec3b>(0);
        uchar* dataR = m.ptr<uchar>(0);
        rgb_to_gs(dataA,dataR,original.rows*original.cols);
    }
    imageColorMode = 1;
    appendImage(m);
    SetImage();
}

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
