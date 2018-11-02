#ifndef STARTTRACKING_H
#define STARTTRACKING_H

#include "cvheaders.h"
#include <QDialog>

enum ModeSourceVideo  { FILE_SOURCE, CAMERA_SOURCE };
enum BackendTracking { CUSTOM_TRACKING, BOOSTRING_TRACKING, MIL_TRACKING, KCF_TRACKING, TLD_TRACKING, MEDIANFLOW_TRACKING, GOTURN_TRACKING, MOSSE_TRACKING, CSRT_TRACKING };
namespace Ui {


class StartTracking;
}

class StartTracking : public QDialog
{
    Q_OBJECT

public:
    explicit StartTracking(QWidget *parent = nullptr);
    ~StartTracking();



private:
    Ui::StartTracking *ui;

    QString inFile;
    QString outFile;
    int		device;
    int		mode;

    int		backend;

    bool	showHSV;
    bool	showTS;

    bool 	filtreNoise;

    bool	record;

    void start();
    void startOpencv();

    cv::VideoCapture getCap();

private slots:
    void onAccept();
    void onReject();

    void onBrowseFileIn();
    void onBrowseFileOut();

    void onRadioBoxChecked();
    void onCheckBoxChecked();
    void on_btnExecuter_clicked();
    void on_btnQuitter_clicked();
    void on_btnParcourirIn_clicked();
};

#endif // STARTTRACKING_H
