#include "tracking.h"
#include "ui_tracking.h"

Tracking::Tracking(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Tracking)
{
    ui->setupUi(this);
}

Tracking::~Tracking()
{
    delete ui;
}
