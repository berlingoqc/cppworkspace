#ifndef MILLEFEUILLEBD_H
#define MILLEFEUILLEBD_H


#include "millefeuille.h"

#include <QString>
#include <QSqlDatabase>
#include <iostream>

class MilleFeuilleBD
{

public:
    static const char* insertNewSourceImg;
    static const char* selectSourceImg;
    static const char* deleteAllSourceImg;
public:
    MilleFeuilleBD();
    bool createConnection();

    bool addSourceImage(const char* filePath, int position, int state);
    bool getSourceImages(Millefeuille_Img_list &images);
    bool deleteAllSourceImage();
private:
    const char* filePath;
    QSqlDatabase db;



};

#endif // MILLEFEUILLEBD_H
