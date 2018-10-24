#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "colorwidget.h"
#include "imagelabel.h"
#include "imagewrapper.h"
#include "cvheaders.h"

#include <QFileDialog>
#include <QStandardPaths>
#include <QMessageBox>
#include <QBoxLayout>
#include <QObject>
#include <sstream>

using namespace cv;



MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    // INITILIAZE les trucs liés avec l'UI
    ui->setupUi(this);

    // Ajout mon label d'image comme widget central
    lblImage = new ImageLabel();
    setCentralWidget(lblImage);


    // Crée mes labels et les ajoutes a ma statusbar
    lblMatInfo = new QLabel(this);
    lblMyImageInfo = new QLabel(this);


    ui->statusBar->addPermanentWidget(lblMatInfo);
    ui->statusBar->addPermanentWidget(lblMyImageInfo);



    // Descative l'option de suivante et precedente
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

    connect(&this->img,&ImageWrapper::imageChanged,this,&MainWindow::updateImage);

    validPreviousNext();
    selectOpencvBackend();

    QString fileName = QStandardPaths::locate(QStandardPaths::HomeLocation,"test.jpg");


    img.appendImageFromFile(fileName.toStdString());

}

void MainWindow::updateStatusBar(const MyImage& img) {
    std::ostringstream m;
    m << "ColorSpace : " << ToString(img.color) << "	Acquision From : " << ToString(img.origin) << " Size : " <<img.image.rows<<"x"<<img.image.cols;
    lblMyImageInfo->setText(tr(m.str().c_str()));
}

void MainWindow::validPreviousNext() {
    bool next = false, previous = false;
    if(img.hasNext()) {
        next = true;
    }
    if (img.hasPrevious()) {
        previous = true;
    }
    ui->actionSuivante->setEnabled(next);
    ui->actionPrecedente->setEnabled(previous);
}

void MainWindow::updateImage(MyImage img) {
    lblImage->SetMat(img.image,img.color);
    validPreviousNext();
    updateStatusBar(img);
}

void MainWindow::loadFileFromCamera() {
    if(!img.appendImageFromCamera(0)) {
        // affiche un message d'erreur si on peut par loader l'image
        QMessageBox msgBox;
        msgBox.setText("Erreur dans la capture depuis la camera");
        msgBox.exec();
    }
}

void MainWindow::loadFileFromFile() {
    QString filePath = QFileDialog::getOpenFileName(this,"Selectionner une image a charger","","JPG (*.jpg);;All Files (*)");
    if(!img.appendImageFromFile(filePath.toStdString())) {
        // affiche un message d'erreur si on peut par loader l'image
        QMessageBox msgBox;
        msgBox.setText("Erreur dans l'ouverture du fichier");
        msgBox.exec();
    }
}

void MainWindow::previousImage() {
    img.previousImage();
}

void MainWindow::nextImage() {
    img.nextImage();
}

void MainWindow::saveImage() {
    QString filePath = QFileDialog::getSaveFileName(this,"Selectionner l'emplacement pour sauvegarder l'i","","JPG (*.jpg);;All Files (*)");
    if(filePath.isEmpty())
        return;
    if(!img.saveCurrentImage(filePath.toStdString())) {
        QMessageBox msgBox;
        msgBox.setText("Erreur dans la sauvegarde du fichier");
        msgBox.exec();
    }
}

void MainWindow::resetOriginalImage() {
    img.returnFirstImage();

}

void MainWindow::closeImage() {
    img.reset();
    validPreviousNext();
    lblImage->clear();
}

void MainWindow::quitApp() {
    this->close();
}

void MainWindow::selectCustomBackend() {
    transformer.setBackend(CustomBackend);
    this->setWindowTitle("Lab4_Vision - Backend : Maison");

}


void MainWindow::selectOpencvBackend() {
    transformer.setBackend(OpencvBackend);
    this->setWindowTitle("Lab4_Vision - Backend : OpenCV");
}

void MainWindow::toHSV() {
    MyImage i = img.getCurrentImage();
    cv::Mat m = i.image.clone();
    transformer.toHSV(i.image,m);
    img.appendImage(m,FromTransformation,HSV_CS);
}

void MainWindow::toBW() {

}

void MainWindow::toGS() {
    MyImage i = img.getCurrentImage();
    if(i.image.empty()) {
        std::cout << "Oups caliss " << std::endl;
    }
    cv::Mat m;
    transformer.toGS(i.image,m);
    img.appendImage(m,FromTransformation,GS_CS);
}

void MainWindow::showHistogramme() {

    MyImage i = img.getCurrentImage();
    if(i.color != RGB_CS) {
        // Si pas RGB on peut pas faire d'histogramme
        showColorError("RGB ou GBR", ToString(i.color));
        return;
    }
    ColorHistogramme histogramme;
    if(!transformer.getColorHistogramme(i.image,&histogramme)) {

    }
    ColorWidget* colorWidget = new ColorWidget(histogramme.Red,histogramme.Green,histogramme.Blue);
    colorWidget->resize(1400,500);
    colorWidget->show();

}
void MainWindow::showDetailImage() {}

void MainWindow::showColorError(QString mustBe,QString is) {
    QMessageBox box;
    box.critical(this,"Erreur",QString().sprintf("Erreur l'image est de couleur %s mais doit être %s pour effectuer cette tâche", is,mustBe));
    box.show();
}

void MainWindow::transformPasseBas() {}
void MainWindow::transformPasseHaut() {}

void MainWindow::transformMedianne()  {}
void MainWindow::transformMoyenne() {}


MainWindow::~MainWindow()
{
    delete ui;
}
