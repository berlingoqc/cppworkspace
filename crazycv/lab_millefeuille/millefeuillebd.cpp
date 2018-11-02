
// import

#include "millefeuillebd.h"


// QT import

#include <QSqlDatabase>
#include <QSqlQuery>
#include <QSqlDriver>
#include <QSqlError>
#include <QSqlRecord>
#include <QVariant>
#include <QMessageBox>
#include <QtDebug>

const char* MilleFeuilleBD::insertNewSourceImg = "INSERT INTO source_img (path,view,state) VALUES (:path,:view,:state)";
const char* MilleFeuilleBD::selectSourceImg = "SELECT path,view,state FROM source_img";
const char* MilleFeuilleBD::deleteAllSourceImg = "DELETE FROM source_img";

MilleFeuilleBD::MilleFeuilleBD()
{
    filePath = "mf.db";
}

bool MilleFeuilleBD::createConnection() {
    db = QSqlDatabase::addDatabase("QSQLITE");
    db.setDatabaseName(filePath);
    if(!db.open()) {
        QMessageBox::critical(nullptr,"Impossible d'ouvrir la bd", "Besoin des drivers SQLite3 QT",QMessageBox::Cancel);
        return false;
    }

    QSqlQuery query(db);
    query.exec("CREATE TABLE IF NOT EXISTS source_img (id INTEGER PRIMARY KEY AUTOINCREMENT, path VARCHAR(100), view INTEGER, state INTEGER)");

    return true;
}

bool MilleFeuilleBD::deleteAllSourceImage() {
    QSqlQuery query(db);
    if(!query.exec(deleteAllSourceImg)) {
        qDebug() << query.lastError();
        return false;
    }
    return true;
}

bool MilleFeuilleBD::addSourceImage(const char* filePath, int position, int state) {
    QSqlQuery query(db);
    query.prepare(insertNewSourceImg);
    query.bindValue( ":path", filePath);
    query.bindValue( ":view", position);
    query.bindValue( ":state", state);
    return query.exec();
}

bool MilleFeuilleBD::getSourceImages(Millefeuille_Img_list& images) {
    QSqlQuery query(db);
    if(!query.exec(selectSourceImg)) {
        qDebug() << query.lastError();
        return false;
    }
    millefeuille_image img;
    while(query.next()) {
        img.filename = query.value(0).toString().toStdString();
        img.view = query.value(1).toInt();
        img.state = query.value(2).toInt();
        images.push_back(img);
    }
    return true;
}
