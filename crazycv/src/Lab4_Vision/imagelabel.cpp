#include "imagelabel.h"

QImage::Format ImageFormatQT[2] { QImage::Format_RGB888, QImage::Format_Grayscale8 };



ImageLabel::ImageLabel(QWidget *parent) :
    QLabel(parent)
{
    setScaledContents(true);
    QSizePolicy policy(QSizePolicy::Maximum, QSizePolicy::Maximum);
    policy.setHeightForWidth(true);
    this->setSizePolicy(policy);
}


void ImageLabel::SetMat(const cv::Mat &m,int colorMode) {
    QImage qI(m.data,m.cols,m.rows, static_cast<int>(m.step), ImageFormatQT[colorMode]);
    QPixmap qPm(QPixmap::fromImage((qI)));
    setPixmap(qPm);
}


void ImageLabel::setPixmap(const QPixmap& pm) {
    pmWidth = pm.width();
    pmHeight = pm.height();

    updateMargins();
    QLabel::setPixmap(pm);
}

void ImageLabel::resizeEvent(QResizeEvent* event) {
    updateMargins();
    QLabel::resizeEvent(event);
}

void ImageLabel::updateMargins(){
    if (pmWidth <= 0 || pmHeight <= 0)
        return;

    int w = this->width();
    int h = this->height();

    if (w <= 0 || h <= 0)
        return;

    if (w * pmHeight > h * pmWidth)
    {
        int m = (w - (pmWidth * h / pmHeight)) / 2;
        setContentsMargins(m, 0, m, 0);
    }
    else
    {
        int m = (h - (pmHeight * w / pmWidth)) / 2;
        setContentsMargins(0, m, 0, m);
    }
}
