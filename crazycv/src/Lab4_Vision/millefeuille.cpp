#include "millefeuille.h"
#include "ui_millefeuille.h"

MilleFeuille::MilleFeuille(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::MilleFeuille)
{
    ui->setupUi(this);
}

MilleFeuille::~MilleFeuille()
{
    delete ui;
}
