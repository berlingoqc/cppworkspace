#ifndef COLORWIDGET_H
#define COLORWIDGET_H

#include <QWidget>
#include <QtCharts/QChartGlobal>

QT_CHARTS_BEGIN_NAMESPACE
class QChartView;
class QChart;
QT_CHARTS_END_NAMESPACE


QT_CHARTS_USE_NAMESPACE


class ColorWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ColorWidget(int* nbrRed,int* nbrBlue,int* nbrGreen, QWidget *parent = nullptr);

signals:

public slots:

private:
    QChart *createAreaChart(QString name,int* data) const;



};

#endif // COLORWIDGET_H
