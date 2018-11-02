#ifndef CONFMILLEFEUILLE_H
#define CONFMILLEFEUILLE_H

#include "cvheaders.h"
#include "millefeuille.h"
#include <QDialog>
#include <QStandardItemModel>



namespace Ui {
class ConfMIlleFeuille;
}

class ConfMIlleFeuille : public QDialog
{
    Q_OBJECT


public:
    explicit ConfMIlleFeuille(QWidget *parent = 0);
    ~ConfMIlleFeuille();

private slots:
    void on_pushButton_4_clicked();

    void on_txtBtnParcourir_clicked();

    void on_btnSauvegarder_clicked();

    void on_btnQuitter_clicked();

    void on_btnAjouter_clicked();

private:
    Ui::ConfMIlleFeuille *ui;
    Millefeuille_Img_list items;

    QStandardItemModel * model;
};

#endif // CONFMILLEFEUILLE_H
