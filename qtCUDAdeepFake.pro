QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    mainwindow.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

win32:CONFIG(release, debug|release): LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_core4100
else:win32:CONFIG(debug, debug|release): LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_core4100
else:unix: LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_core4100


win32:CONFIG(release, debug|release): LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_cudaobjdetect4100
else:win32:CONFIG(debug, debug|release): LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_cudaobjdetect4100
else:unix: LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_cudaobjdetect4100

win32:CONFIG(release, debug|release): LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_videoio4100
else:win32:CONFIG(debug, debug|release): LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_videoio4100
else:unix: LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_videoio4100


win32:CONFIG(release, debug|release): LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_imgcodecs4100
else:win32:CONFIG(debug, debug|release): LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_imgcodecs4100
else:unix: LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_imgcodecs4100

win32:CONFIG(release, debug|release): LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_highgui4100
else:win32:CONFIG(debug, debug|release): LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_highgui4100
else:unix: LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_highgui4100

win32:CONFIG(release, debug|release): LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_imgproc4100
else:win32:CONFIG(debug, debug|release): LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_imgproc4100
else:unix: LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_imgproc4100

win32:CONFIG(release, debug|release): LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_face4100
else:win32:CONFIG(debug, debug|release): LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_face4100
else:unix: LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_face4100


win32:CONFIG(release, debug|release): LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_cudaimgproc4100
else:win32:CONFIG(debug, debug|release): LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_cudaimgproc4100
else:unix: LIBS += -L'D:/opencv 4.10 extra install/install/x64/vc17/lib/' -lopencv_cudaimgproc4100

INCLUDEPATH += 'D:/opencv 4.10 extra install/install/include'
DEPENDPATH += 'D:/opencv 4.10 extra install/install/include'
