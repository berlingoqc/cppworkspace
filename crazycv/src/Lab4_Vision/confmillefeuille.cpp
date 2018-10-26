#include "confmillefeuille.h"
#include "ui_confmillefeuille.h"

#include <QMessageBox>
#include <QStandardItemModel>
#include <QtSql/QSqlDatabase>
#include <QtSql/QSqlTableModel>
#include <QtSql/QSqlQuery>

bool ConfMIlleFeuille::createConnection() {
    QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE");
    db.setDatabaseName(":memory:");
    if(!db.open()) {
        QMessageBox::critical(nullptr,"Impossible d'ouvrir la bd", "Besoin des drivers SQLite3 QT",QMessageBox::Cancel);
        return false;
    }

    QSqlQuery query;
    query.exec("create table if not exists source_img(id int primary key, path varchar(100), view integer, error integer");
    query.exec("insert into source_img(1,'/home/wq/mille_feuille_1.jpg',0,0");

    return true;
}

ConfMIlleFeuille::ConfMIlleFeuille(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ConfMIlleFeuille)
{
    ui->setupUi(this);


    // Crée notre base de donnée
    createConnection();
    model  = new QSqlTableModel(this);
    model->setTable("source_img");
    model->setEditStrategy(QSqlTableModel::OnManualSubmit);
    model->select();

    model->setHeaderData(0,Qt::Horizontal, "ID");
    model->setHeaderData(1,Qt::Horizontal, "Image");
    model->setHeaderData(2,Qt::Horizontal, "Vue");
    model->setHeaderData(3,Qt::Horizontal, "Erreur");

    ui->table->setModel(model);
    ui->table->resizeColumnsToContents();

    connect(ui->btnSauvegarder, &QPushButton::clicked, model, &QSqlTableModel::submit);

    connect(ui->btnQuitter, &QPushButton::clicked, model, &QSqlTableModel::clear);

}

ConfMIlleFeuille::~ConfMIlleFeuille()
{
    delete ui;
}

void ConfMIlleFeuille::on_pushButton_4_clicked()
{

}

void ConfMIlleFeuille::on_txtBtnParcourir_clicked()
{

}

void ConfMIlleFeuille::on_btnSauvegarder_clicked()
{

}

void ConfMIlleFeuille::on_btnQuitter_clicked()
{

}
