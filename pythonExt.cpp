
#include "FaceDetector.h"
#include "FaceRecognition.h"
//#include <string>
extern "C" {
    FaceRecognition *faceRecognition_public = nullptr;

    long InitFaceSession(char* model_filename)
    {
        std::string filename(model_filename);
        FaceRecognition *faceRecognition = new FaceRecognition(filename);
        faceRecognition_public = faceRecognition;
        return  (long)faceRecognition;
    }


    float* ExtractorFeatures(long session,char* image,int w,int h)
    {
        cv::Mat image_out(h,w,CV_8UC3,image);
        FaceRecognition *session_instance = (FaceRecognition *)session;
        cv::Mat output;
        session_instance->extractSingleFaceFeatures(image_out,output);
        return (float*)output.data;
    }


    float* ExtractorFeatures_direct(long session,char* image,int w,int h)
    {
        cv::Mat transformed(h,w,CV_8UC3,image);
        FaceRecognition *session_instance = (FaceRecognition *)session;
        Mat blob_input_ = dnn::blobFromImage(transformed, 0.0078125, cv::Size(112, 112),
                                             cv::Scalar(127.5, 127.5, 127.5), true );
        session_instance->frNet.setInput(blob_input_);
        cv::Mat res = session_instance->frNet.forward();
        return (float*)res.data;
    }

    void MtcnnFaceDetector(long session,char* image, int w,int h,int minSize,float** output,int* outputNum)
    {
        const float threshold[3] = {0.7f, 0.6f, 0.6f};
        float factor = 0.709;
        cv::Mat image_out(h,w,CV_8UC3,image);
         FaceRecognition *session_instance = (FaceRecognition *)session;
        vector<FaceInfo> faceInfo = session_instance->faceDetector->Detect_mtcnn(image_out, minSize, threshold, factor, 3);
        *outputNum = faceInfo.size();
        *output = new float[faceInfo.size()*15];
        float *data = *output;
        for(int i = 0 ; i < faceInfo.size() ; i++)
        {
            data[i*15 + 0 ] = faceInfo[i].bbox.score;
            data[i*15 + 1 ] = faceInfo[i].bbox.xmin;
            data[i*15 + 2 ] = faceInfo[i].bbox.ymin;
            data[i*15 + 3 ] = faceInfo[i].bbox.xmax;
            data[i*15 + 4 ] = faceInfo[i].bbox.ymax;
            data[i*15 + 5] = faceInfo[i].landmark[0];
            data[i*15 + 6] = faceInfo[i].landmark[1];
            data[i*15 + 7] = faceInfo[i].landmark[2];
            data[i*15 + 8] = faceInfo[i].landmark[3];
            data[i*15 + 9] = faceInfo[i].landmark[4];
            data[i*15 + 10] = faceInfo[i].landmark[5];
            data[i*15 + 11] = faceInfo[i].landmark[6];
            data[i*15 + 12] = faceInfo[i].landmark[7];
            data[i*15 + 13] = faceInfo[i].landmark[8];
            data[i*15 + 14] = faceInfo[i].landmark[9];
        }
    }

    void ReleaseFaceSession(long session)
    {
        FaceRecognition *faceRecognition = (FaceRecognition *)session;
        delete faceRecognition;
    }

    void FreeFloat(float** memblock)
    {
        std::cout<<*memblock<<std::endl;

        delete []*memblock;

    }

    void FreeFloat1D(float** memblock)
    {
        std::cout<<*memblock<<std::endl;


        delete []*memblock;

    }


}


