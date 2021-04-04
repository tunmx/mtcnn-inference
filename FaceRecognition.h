//
// Created by Jack Yu on 23/03/2018.
//

#ifndef FACE_DEMO_FACERECOGNITION_H
#define FACE_DEMO_FACERECOGNITION_H

#include "FaceDetector.h"
#include "ModelLoader.h"
#include "FacePreprocess.h"

class FaceRecognition{

public:

    ModelReader::ModelLoader *model;
    MTCNN *faceDetector;
    dnn::Net frNet;
        FaceRecognition(const std::string filenameModel)
        {
            model = new ModelReader::ModelLoader(filenameModel);
            faceDetector = new MTCNN(model);
            ModelReader::Model *modelFR =model->readModel(3);
            frNet = cv::dnn::readNetFromCaffe(modelFR->prototxtBuffer, modelFR->modelsize.prototxt_size,modelFR->caffemodelBuffer,modelFR->modelsize.caffemodel_size);
        }
        cv::Mat extractSingleFaceFeatures(const cv::Mat &image,cv::Mat &outputVector) {
            const float factor = 0.709f;
            const float threshold[3] = {0.7f, 0.6f, 0.6f};
            const int minSize = 50;
            vector<FaceInfo> faceInfo = faceDetector->Detect_mtcnn(image, minSize, threshold, factor, 3);
            std::vector<std::pair<cv::Mat, cv::Rect>> features;
            if (faceInfo.size() != 1) {
                std::cout << "faceInfo.size()!=1:" << std::endl;
                std::cout << "please ensure only one face in your image." << std::endl;
                assert(faceInfo.size() != 1);
                return cv::Mat();
            }
            cv::Size image_size(112, 112);
            int i = 0;
            int x = (int) faceInfo[i].bbox.xmin;
            int y = (int) faceInfo[i].bbox.ymin;
            int x1 = (int) (faceInfo[i].bbox.xmax);
            int y1 = (int) (faceInfo[i].bbox.ymax);
            int w = (int) (faceInfo[i].bbox.xmax - faceInfo[i].bbox.xmin + 1);
            int h = (int) (faceInfo[i].bbox.ymax - faceInfo[i].bbox.ymin + 1);
            float *landmark = faceInfo[i].landmark;
            cv::Size bounding_box_size(w, h);
            float src_pts[] = {30.2946, 52.6963,
                               65.5318, 52.5014,
                               48.0252, 71.7366,
                               33.5493, 92.3655,
                               62.7299, 92.2041};
            if (image_size.height) {
                for (int i = 0; i < 5; i++) {
                    *(src_pts + 2 * i) += 8.0;
                }

            }
            cv::Mat src(5, 2, CV_32F);
            src.data = (uchar *) src_pts;
            cv::Mat dst(5, 2, CV_32F);
            dst.data = (uchar *) landmark;
            cv::Mat M_temp = FacePreprocess::similarTransform(dst, src);
            cv::Mat M = M_temp.rowRange(0, 2);
            cv::Mat transformed;
            cv::warpAffine(image, transformed, M, cv::Size(112, 112),cv::INTER_CUBIC);
            Mat blob_input_ = dnn::blobFromImage(transformed, 0.0078125, cv::Size(112, 112),
                                                cv::Scalar(127.5, 127.5, 127.5), true );
            frNet.setInput(blob_input_);
            cv::Mat res = frNet.forward();
            res.copyTo(outputVector);
            return res;

        }

    inline float CosineDistance(const cv::Mat &v1,const cv::Mat &v2){
            return (v1.dot(v2));
    }


    //
    float compareFace(const cv::Mat &imageA,const cv::Mat &imageB)
    {

        cv::Mat v1;
        extractSingleFaceFeatures(imageA,v1);
        cv::Mat v2;
        extractSingleFaceFeatures(imageB,v2);
        cv::Mat f1,f2;
        cv::multiply(v1,v1,f1);
        cv::multiply(v2,v2,f2);
        float norm_f1 = cv::sqrt(cv::sum(f1).val[0]);
        float norm_f2 = cv::sqrt(cv::sum(f2).val[0]);
        v1 = v1 / norm_f1;
        v2 = v2 / norm_f2;
        return CosineDistance(v1,v2);

    }

    ~FaceRecognition(){

        delete faceDetector;
        delete model;

    }









};

#endif //FACE_DEMO_FACERECOGNITION_H
