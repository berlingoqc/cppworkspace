#include "confmillefeuille.h"
#include "ui_confmillefeuille.h"

#include <QMessageBox>
#include <QStandardItemModel>
#include <QtSql/QSqlDatabase>
#include <QtSql/QSqlTableModel>
#include <QtSql/QSqlQuery>



ConfMIlleFeuille::ConfMIlleFeuille(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ConfMIlleFeuille)
{
    ui->setupUi(this);


    // Crée notre base de donnée
    model  = new QStandardItemModel();

    model->setHeaderData(0,Qt::Horizontal, "ID");
    model->setHeaderData(1,Qt::Horizontal, "Image");
    model->setHeaderData(2,Qt::Horizontal, "Vue");
    model->setHeaderData(3,Qt::Horizontal, "Erreur");

    ui->table->setModel(model);
    ui->table->resizeColumnsToContents();

}

ConfMIlleFeuille::~ConfMIlleFeuille()
{
    delete ui;
}




void ConfMIlleFeuille::on_txtBtnParcourir_clicked()
{
    // Parcour pour une image

}

void ConfMIlleFeuille::on_btnSauvegarder_clicked()
{
    // Ferme le dialog

}

void ConfMIlleFeuille::on_btnQuitter_clicked()
{

}

void ConfMIlleFeuille::on_btnAjouter_clicked()
{

}
