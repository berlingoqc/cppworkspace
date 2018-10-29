#ifndef TRACKING_H
#define TRACKING_H

#include <QDialog>

namespace Ui {
class Tracking;
}

class Tracking : public QDialog
{
    Q_OBJECT

public:
    explicit Tracking(QWidget *parent = nullptr);
    ~Tracking();

private:
    Ui::Tracking *ui;
};

#endif // TRACKING_H
