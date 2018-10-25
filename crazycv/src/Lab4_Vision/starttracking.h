#ifndef STARTTRACKING_H
#define STARTTRACKING_H

#include <QDialog>

enum ModeSourceVideo  { FILE_SOURCE, CAMERA_SOURCE };

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

    bool	showHSV;
    bool	showTS;

    bool 	filtreNoise;

    bool	record;

    void start();

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
