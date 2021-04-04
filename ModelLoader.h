#ifndef SWIFTPR_MODELLOADER_H
#define SWIFTPR_MODELLOADER_H

#include <string>
#include <fstream>
#include <iostream>
#include "opencv2/dnn.hpp"



namespace ModelReader{

    struct ModelSize{
        int prototxt_size;
        int caffemodel_size;
    };


    struct Model{
        char *prototxtBuffer;
        char *caffemodelBuffer;
        ModelSize modelsize;

    };


    class ModelLoader{
    public:
        ModelLoader(std::string model_path);
        ~ModelLoader();
        Model* readModel(int idx);


    private:

        ModelSize *modelSize;
        Model *model;
        int number_of_models;
        int magic_number ;

    };
} //namespace pr


#endif //SWIFTPR_MODELLOADER_H
