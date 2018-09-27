#ifndef IMAGELABEL_H
#define IMAGELABEL_H

#include "cvheaders.h"
#include <QLabel>


class ImageLabel : public QLabel
{
    Q_OBJECT
public:
    explicit ImageLabel(QWidget *parent = nullptr);

    void SetMat(const cv::Mat &m);
public slots:
    void setPixmap(const QPixmap& pm);
protected:
    void resizeEvent(QResizeEvent* event) override;
private:
    void updateMargins();

    int pmWidth = 0;
    int pmHeight = 0;

};

#endif // IMAGELABEL_H
