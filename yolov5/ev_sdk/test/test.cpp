#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <onnxruntime_cxx_api.h>  
#include <cuda_provider_factory.h>
#include "Yolov5.hpp"

int main()
{
    cv::Mat src = cv::imread("./data/bus.jpg");  //默认uchar 
    std::string modelpath = "./model/yolov5s.onnx";
    std::string classesFile = "./model/class.txt";
    if (src.empty())
    {
        printf("faied to load image ...\n");
    }
    else
    {       
        printf("\ninference...\n");
        std::vector<Yolov5::Object> validTargets;
        auto *yolov5 = new Yolov5(modelpath,classesFile,0.3,0.5,0.3);
        yolov5->forward(src,validTargets);

        cv::imwrite("out.jpg", src);
        printf("done!\n");
    }
    return 0;
}


