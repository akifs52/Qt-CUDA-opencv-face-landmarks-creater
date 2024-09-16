#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/videoio.hpp>
#include <QCoreApplication>
#include <opencv2/imgproc.hpp>
QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_start_clicked();

    void updateFrame();

    void on_close_clicked();

    void on_browse_clicked();
    //landmarkları birleşitirip üçgen yap
    void drawDelaunay(cv::Mat& img, cv::Subdiv2D& subdiv, cv::Scalar delaunayColor);

private:
    Ui::MainWindow *ui;


    cv::Ptr<cv::Mat> selectedCPUimage;
    cv::Ptr<cv::cuda::GpuMat> selectedGPUImage;
    cv::Ptr<cv::cuda::CascadeClassifier> faceCascade;

    QString yamlPath = QCoreApplication::applicationDirPath() + ("/resources/quality/lbfmodel.yaml");
    QString lbpPath = QCoreApplication::applicationDirPath() + ("/resources/lbpcascades/lbpcascade_frontalface_improved.xml");
    QTimer *timer;



    cv::Ptr<cv::VideoCapture> cap;

};
#endif // MAINWINDOW_H
