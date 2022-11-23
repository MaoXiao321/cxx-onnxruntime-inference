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

cv::Mat outputFrame{ 0 };        // ���ڴ洢�㷨���������ͼ�񣬸���ji.h�Ľӿڹ淶���ӿ�ʵ����Ҫ�����ͷŸ���Դ
//ʵ��ji_create_predictor��ji_calc_frame��processMat����


int processMat(SampleDetector* detector, const cv::Mat& inFrame, const char* args, cv::Mat& outFrame, JI_EVENT& event) {
    std::vector<SampleDetector::Object> detectedObjects;
    std::vector<SampleDetector::Object> validTargets;

    int processRet = detector->detect(inFrame, validTargets);  // �㷨�����ĳ��Լ��Ĵ�������
    if (processRet != SampleDetector::PROCESS_OK) {
        return JISDK_RET_FAILED;
    }
    static const string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL); //ָ������������
    imshow(kWinName, inFrame); //��ʾ�����
    waitKey(0);
    destroyAllWindows();
    return JISDK_RET_SUCCEED;
}


void* ji_create_predictor(int pdtype) {
    //��һ��init
    auto* detector = new SampleDetector(algoConfig.thresh, algoConfig.nms, algoConfig.hierThresh);
    int iRet = detector->init("../class.names", "../yolov5s.onnx");
    return detector;
}

int ji_calc_frame(void* predictor, const JI_CV_FRAME* inFrame, const char* args,
    //���ø�
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
