#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QTimer>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/face.hpp>
#include <opencv2/highgui.hpp>
#include <QDebug>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , selectedCPUimage(new cv::Mat())
    , selectedGPUImage(new cv::cuda::GpuMat())
    , cap(new cv::VideoCapture())
{
    ui->setupUi(this);

    timer = new QTimer(this);

    try {

        faceCascade = faceCascade->create(lbpPath.toStdString());
    } catch (cv::Exception &e) {

        qDebug() << "OpenCV hatası: " << e.what();
    }

}

MainWindow::~MainWindow()
{
    cap.release();
    disconnect(timer,&QTimer::timeout,this,&MainWindow::updateFrame);
    delete ui;
}

void MainWindow::on_start_clicked()
{
    try {
        cap->open(0);
        if (!cap->isOpened()) {
            QMessageBox::warning(this, "Camera is not opening", "Check your cameras");
            return;
        }

        timer->start(30); // 30ms
        connect(timer, &QTimer::timeout,this,&MainWindow::updateFrame);
    } catch (const cv::Exception &e) {
        qDebug() << "OpenCV hatası: " << e.what();
    } catch (const std::exception &e) {
        qDebug() << "Genel hata: " << e.what();
    } catch (...) {
        qDebug() << "Bilinmeyen bir hata oluştu.";
    }

}

void MainWindow::updateFrame()
{
    try {

        // GpuMat ve CPU Mat nesneleri
        cv::Ptr<cv::cuda::GpuMat> gpuFrame = cv::makePtr<cv::cuda::GpuMat>();
        cv::Ptr<cv::Mat> frameCPU = cv::makePtr<cv::Mat>();

        // Kameradan bir frame al
        cv::Ptr<cv::Mat> frameBGR = cv::makePtr<cv::Mat>();
        if (cap->read(*frameBGR)) {
            // BGR frame'i GPU'ya yükle
            gpuFrame->upload(*frameBGR);

            // GPU üzerinde BGR'den RGB'ye dönüşüm
            cv::cuda::cvtColor(*gpuFrame, *gpuFrame, cv::COLOR_BGR2RGB);

            // GPU üzerinde RGB'den Grayscale'e dönüşüm
            cv::Ptr<cv::cuda::GpuMat> gpuGrayFrame = cv::makePtr<cv::cuda::GpuMat>();
            cv::cuda::cvtColor(*gpuFrame, *gpuGrayFrame, cv::COLOR_RGB2GRAY);

            // Yüz tespiti
            cv::Ptr<cv::cuda::GpuMat> gpuFrameMemory = cv::makePtr<cv::cuda::GpuMat>();
            faceCascade->detectMultiScale(*gpuGrayFrame, *gpuFrameMemory);

            std::vector<cv::Rect> faces;
            faceCascade->convert(*gpuFrameMemory, faces);
            cv::Mat facesMat(faces.size(), 1, CV_32SC4, &faces[0]);

            cv::Ptr<cv::face::Facemark> facemark = cv::face::FacemarkLBF::create();
            facemark->loadModel(yamlPath.toStdString());

            // GPU'dan CPU'ya indirme
            gpuFrame->download(*frameCPU);

            std::vector<std::vector<cv::Point2f>> landmarks;
            if (facemark->fit(*frameCPU, facesMat, landmarks)) {
                for (size_t i = 0; i < landmarks.size(); i++) {
                    // Delaunay triangulation için yüz çerçevesini hazırlayın
                    cv::Rect rect(0, 0, frameCPU->cols, frameCPU->rows);
                    cv::Subdiv2D subdiv(rect);

                    // Tüm landmark noktalarını subdiv'e ekleyin
                    for (size_t j = 0; j < landmarks[i].size(); j++) {
                        subdiv.insert(landmarks[i][j]);
                        cv::circle(*frameCPU, landmarks[i][j], 3, cv::Scalar(0, 0, 255), -1);
                    }

                    // Delaunay triangulation çizimi
                    drawDelaunay(*frameCPU, subdiv, cv::Scalar(255, 255, 255));
                }
            }

            // OpenCV Mat'ı QImage'e dönüştür
            QImage img(frameCPU->data, frameCPU->cols, frameCPU->rows, frameCPU->step, QImage::Format_RGB888);

            // QLabel üzerinde görüntüyü göster
            ui->label->setPixmap(QPixmap::fromImage(img));
            ui->label->setScaledContents(true);
        }
    } catch (const cv::Exception &e) {
        qDebug() << "OpenCV hatası: " << e.what();
    } catch (const std::exception &e) {
        qDebug() << "Genel hata: " << e.what();
    } catch (...) {
        qDebug() << "Bilinmeyen bir hata oluştu.";
    }
}

void MainWindow::on_close_clicked()
{
    cap->release();
    disconnect(timer,&QTimer::timeout,this,&MainWindow::updateFrame);
}


void MainWindow::on_browse_clicked()
{
    try {
        QString fileName = QFileDialog::getOpenFileName(this, "Choose a picture", "", "Images (*.png *.xpm *.jpg)");

        if (!fileName.isEmpty()) {
            // CPU'dan GPU'ya görüntüyü aktarma
            *selectedCPUimage = cv::imread(fileName.toStdString());
            if (selectedCPUimage->empty()) {
                throw std::runtime_error("Görüntü yüklenemedi.");
            }

            // Convert to grayscale
            cv::Mat grayImage;
            cv::cvtColor(*selectedCPUimage, grayImage, cv::COLOR_BGR2GRAY);

            // Ensure the image type is CV_8UC1
            if (grayImage.type() != CV_8UC1) {
                throw std::runtime_error("Görüntü tipi CV_8UC1 değil.");
            }

            selectedGPUImage->upload(grayImage);

            // Yüz matrislerini koordinatını tutar
            cv::Ptr<cv::cuda::GpuMat> selectedImageMemory = cv::makePtr<cv::cuda::GpuMat>();

            // Yüzü tespit eder
            faceCascade->detectMultiScale(*selectedGPUImage, *selectedImageMemory);

            std::vector<cv::Rect> faces;
            faceCascade->convert(*selectedImageMemory, faces);
            cv::Mat facesMat(faces.size(), 1, CV_32SC4, &faces[0]);

            cv::Ptr<cv::face::Facemark> facemark = cv::face::FacemarkLBF::create();
            facemark->loadModel(yamlPath.toStdString());

            std::vector<std::vector<cv::Point2f>> landmarks;
            if (facemark->fit(*selectedCPUimage, facesMat, landmarks)) {
                for (size_t i = 0; i < landmarks.size(); i++) {
                    // Delaunay triangulation için yüz çerçevesini hazırlayın
                    cv::Rect rect(0, 0, selectedCPUimage->cols, selectedCPUimage->rows);
                    cv::Subdiv2D subdiv(rect);

                    // Tüm landmark noktalarını subdiv'e ekleyin
                    for (size_t j = 0; j < landmarks[i].size(); j++) {
                        subdiv.insert(landmarks[i][j]);
                        cv::circle(*selectedCPUimage, landmarks[i][j], 3, cv::Scalar(0, 0, 255), -1);
                    }

                    // Delaunay triangulation çizimi
                    drawDelaunay(*selectedCPUimage, subdiv, cv::Scalar(255, 255, 255));
                }
            }
        }
        cv::imshow("landmarks and triangulation", *selectedCPUimage);
    } catch (const cv::Exception& e) {
        qDebug() << "OpenCV hatası: " << e.what();
    } catch (const std::exception& e) {
        qDebug() << "Genel hata: " << e.what();
    } catch (...) {
        qDebug() << "Bilinmeyen bir hata oluştu.";
    }
}


void MainWindow::drawDelaunay(cv::Mat& img, cv::Subdiv2D& subdiv, cv::Scalar delaunayColor) {
    std::vector<cv::Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    std::vector<cv::Point> pt(3);
    cv::Size size = img.size();
    cv::Rect rect(0, 0, size.width, size.height);

    for (size_t i = 0; i < triangleList.size(); i++) {
        cv::Vec6f t = triangleList[i];
        pt[0] = cv::Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = cv::Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = cv::Point(cvRound(t[4]), cvRound(t[5]));

        // Check if the triangle is inside the image
        if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
            line(img, pt[0], pt[1], delaunayColor, 1, cv::LINE_AA, 0);
            line(img, pt[1], pt[2], delaunayColor, 1, cv::LINE_AA, 0);
            line(img, pt[2], pt[0], delaunayColor, 1, cv::LINE_AA, 0);
        }
    }
}
