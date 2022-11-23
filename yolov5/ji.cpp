#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "ji.h"
#include "SampleDetector.hpp"

using namespace std;
using ALGO_CONFIG_TYPE = typename SampleDetector::ALGO_CONFIG_TYPE;
ALGO_CONFIG_TYPE algoConfig{ 0.6, 0.5, 0.5 };

cv::Mat outputFrame{ 0 };        // 用于存储算法处理后的输出图像，根据ji.h的接口规范，接口实现需要负责释放该资源
//实现ji_create_predictor、ji_calc_frame和processMat即可


int processMat(SampleDetector* detector, const cv::Mat& inFrame, const char* args, cv::Mat& outFrame, JI_EVENT& event) {
    std::vector<SampleDetector::Object> detectedObjects;
    std::vector<SampleDetector::Object> validTargets;

    int processRet = detector->detect(inFrame, validTargets);  // 算法处理，改成自己的处理函数名
    if (processRet != SampleDetector::PROCESS_OK) {
        return JISDK_RET_FAILED;
    }
    static const string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL); //指定弹窗的名称
    imshow(kWinName, inFrame); //显示检测结果
    waitKey(0);
    destroyAllWindows();
    return JISDK_RET_SUCCEED;
}


void* ji_create_predictor(int pdtype) {
    auto* detector = new SampleDetector(algoConfig.thresh, algoConfig.nms, algoConfig.hierThresh);
    int iRet = detector->init("../class.names", "../yolov5s.onnx");
    return detector;
}

int ji_calc_frame(void* predictor, const JI_CV_FRAME* inFrame, const char* args,
    //不用改
    JI_CV_FRAME* outFrame, JI_EVENT* event) {
    if (predictor == NULL || inFrame == NULL) {
        return JISDK_RET_INVALIDPARAMS;
    }
    auto* detector = reinterpret_cast<SampleDetector*>(predictor);
    cv::Mat inMat(inFrame->rows, inFrame->cols, inFrame->type, inFrame->data, inFrame->step);
    int processRet = processMat(detector, inMat, args, outputFrame, *event);

    if (processRet == JISDK_RET_SUCCEED) {
        if ((event->code != JISDK_CODE_FAILED) && (!outputFrame.empty()) && (outFrame)) {
            outFrame->rows = outputFrame.rows;
            outFrame->cols = outputFrame.cols;
            outFrame->type = outputFrame.type();
            outFrame->data = outputFrame.data;
            outFrame->step = outputFrame.step;
        }
    }
    cout << endl;
    cout << "done!" << endl;
    return processRet;
}
