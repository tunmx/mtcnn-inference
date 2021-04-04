#include "ModelLoader.h"

namespace ModelReader{

    ModelLoader::ModelLoader(std::string model_path) {
        std::ifstream file(model_path, std::ios::binary);
        magic_number = 0;
        number_of_models = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        if (magic_number != 1127)
        {
            assert(magic_number != 1127);
        }
        file.read((char*)&number_of_models, sizeof(number_of_models));
        modelSize  = new ModelSize[number_of_models];
        file.read((char*)modelSize, sizeof(ModelSize)*number_of_models);
        model = new Model[number_of_models];
        std::cout<<"magic_number:"<<magic_number<<std::endl;
        std::cout<<"number_of_models:"<<number_of_models<<std::endl;

        for(int i = 0 ; i < number_of_models ; i ++)
        {
            model[i].prototxtBuffer = new char[modelSize[i].prototxt_size];
            model[i].caffemodelBuffer = new char[modelSize[i].caffemodel_size];
            model[i].modelsize = modelSize[i];
            file.read(model[i].prototxtBuffer, modelSize[i].prototxt_size);
            file.read(model[i].caffemodelBuffer, modelSize[i].caffemodel_size);

            std::cout<<i<<std::endl;
            std::cout<<model[i].modelsize.prototxt_size<<std::endl;
            std::cout<<model[i].modelsize.caffemodel_size<<std::endl;

        }

    }

    ModelLoader::~ModelLoader() {
        //release memory
        for(int i = 0 ; i < number_of_models ; i ++)
        {

            delete model[i].prototxtBuffer;
            delete model[i].caffemodelBuffer;
        }
        delete model;
        delete modelSize;

    }

    Model* ModelLoader::readModel(int idx) {
        return &model[idx];
    }



}