#include "../inc/home.h"
#include "ui_home.h"

home::home(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::home)
{
    ui->setupUi(this);
}

home::~home()
{
    delete ui;
}

void home::on_btnTP3_clicked()
{

}

void home::on_btnTP4_clicked()
{

}

void home::on_btnTP5_clicked()
{

}
