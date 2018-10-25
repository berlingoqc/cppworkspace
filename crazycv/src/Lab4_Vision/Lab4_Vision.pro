#-------------------------------------------------
#
# Project created by QtCreator 2018-09-27T16:32:45
#
#-------------------------------------------------

QT       += core gui charts

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Lab4_Vision
TEMPLATE = app

QMAKE_CFLAGS += -std=c99

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++11

SOURCES += \
        main.cpp \
        mainwindow.cpp \
    imagelabel.cpp \
    colorwidget.cpp \
    imagewrapper.cpp \
    starttracking.cpp

HEADERS += \
        mainwindow.h \
    cvheaders.h \
    imagelabel.h \
    colorwidget.h \
    imagewrapper.h \
    starttracking.h

FORMS += \
        mainwindow.ui \
    starttracking.ui

win32 {
        LIBS += -LC:/msys64/mingw64/bin -lopencv_core342 -lopencv_videoio342 -lopencv_imgproc342 -lopencv_highgui342 -lopencv_imgcodecs342
}
!win32 {
        LIBS += -lopencv_world
}
# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
