#include "colorwidget.h"

#include <QtCharts/QChartView>
#include <QtCharts/QAreaSeries>
#include <QtCharts/QLineSeries>
#include <QtCharts/QLegend>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>
#include <QtCore/QTime>
#include <QtCharts/QBarCategoryAxis>

ColorWidget::ColorWidget(int* nbrRed,int* nbrBlue,int* nbrGreen, QWidget *parent) : QWidget(parent)
{

    // Crée notre layout de base
    QGridLayout *baseLayout = new QGridLayout();

    //Crée nos trois chartes pour les trois couleurs
    QChartView *chartView;

    chartView = new QChartView(createAreaChart("Rouge",nbrRed));
    baseLayout->addWidget(chartView,1,0);
    chartView->setRenderHint(QPainter::Antialiasing,true);

    chartView = new QChartView(createAreaChart("Bleue",nbrBlue));
    baseLayout->addWidget(chartView,1,1);
    chartView->setRenderHint(QPainter::Antialiasing,true);

    chartView = new QChartView(createAreaChart("Vert",nbrGreen));
    baseLayout->addWidget(chartView,1,2);
    chartView->setRenderHint(QPainter::Antialiasing,true);

    setLayout(baseLayout);




}

QChart* ColorWidget::createAreaChart(const QString name,int* data) const {
    QChart* chart = new QChart();
    chart->setTitle(name);

    // Definit la lower series
    QLineSeries *lowerSeries = 0;
    QString nameS("Series ");

    QLineSeries *upperSeries = new QLineSeries(chart);

    for(int i=0;i<256;i++) {
        if (lowerSeries) {
            const QVector<QPointF>& points = lowerSeries->pointsVector();
            upperSeries->append(QPointF(i, points[i].y() + data[i]));
        } else {
            upperSeries->append(QPointF(i, data[i]));
        }
    }

    QAreaSeries *area = new QAreaSeries(upperSeries,lowerSeries);
    area->setName(name+" area");
    chart->addSeries(area);
    chart->createDefaultAxes();
    chart->setAnimationOptions(QChart::AllAnimations);

    return chart;
}
