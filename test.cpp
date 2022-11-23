#include <string>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "ji.h"
using namespace std;

// 没有要改的

void* predictor = NULL;
string strIn = "../bus.jpg";

//test.cpp不需要修改
int main(int argc, char* argv[]) {
    predictor = ji_create_predictor(JISDK_PREDICTOR_DEFAULT);
    if (predictor == NULL)
    {
        cout << "[ERROR] ji_create_predictor faild, return NULL";
        return -1;
    }
    JI_CV_FRAME inframe, outframe;
    JI_EVENT event;
    cv::Mat inMat = cv::imread(strIn);
    if (inMat.empty())
    {
        cout << "[ERROR] cv::imread source file failed, " << strIn;
        return -1;
    }
    inframe.rows = inMat.rows;
    inframe.cols = inMat.cols;
    inframe.step = inMat.step;
    inframe.data = inMat.data;
    inframe.type = inMat.type();
    int iRet;
    iRet = ji_calc_frame(predictor, &inframe, "", &outframe, &event);
    return iRet;
}