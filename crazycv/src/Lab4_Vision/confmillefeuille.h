#ifndef CONFMILLEFEUILLE_H
#define CONFMILLEFEUILLE_H

#include "cvheaders.h"
#include <QDialog>
#include <QStandardItemModel>
#include <QtSql/QSqlTableModel>
#include <QItemDelegate>

enum MilleFeuilleView { TOP_VIEW, SIDE_VIEW };
enum MilleFeuilleError { NOT_CRUSTAD_ERROR, NOT_TOP_ERROR, NOT_CREME_ERROR };

struct millefeuille_image {
    cv::Mat 	img;
    std::string filename;
    int 		view;
    int			error;
};

namespace Ui {
class ConfMIlleFeuille;
}

class ConfMIlleFeuille : public QDialog
{
    Q_OBJECT

public:
    bool createConnection();

public:
    explicit ConfMIlleFeuille(QWidget *parent = 0);
    ~ConfMIlleFeuille();

private slots:
    void on_pushButton_4_clicked();

    void on_txtBtnParcourir_clicked();

    void on_btnSauvegarder_clicked();

    void on_btnQuitter_clicked();

private:
    Ui::ConfMIlleFeuille *ui;
    QSqlTableModel * model;
};

#endif // CONFMILLEFEUILLE_H
