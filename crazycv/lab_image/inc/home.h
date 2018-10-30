#ifndef HOME_H
#define HOME_H

#include <QDialog>

namespace Ui {
class home;
}

class home : public QDialog
{
    Q_OBJECT

public:
    explicit home(QWidget *parent = nullptr);
    ~home();

private slots:
    void on_btnTP3_clicked();

    void on_btnTP4_clicked();

    void on_btnTP5_clicked();

private:
    Ui::home *ui;
};

#endif // HOME_H
