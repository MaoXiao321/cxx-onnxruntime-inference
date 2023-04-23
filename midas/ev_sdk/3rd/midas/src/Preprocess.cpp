#include <opencv2/opencv.hpp>
#include <iostream>
#include "Preprocess.hpp"


Preprocess::Preprocess(int height, int width, cv::Scalar mean, std::vector<float> std, bool keep_aspect_ratio)
{
    __height = height;
    __width =  width;
    __mean = mean;
    __std = std;
    __keep_aspect_ratio = keep_aspect_ratio;
}

void Preprocess::resize(cv::Mat image, cv::Mat &out, std::vector<int> &pad)
{
    if (__keep_aspect_ratio)
    {
        float input_h = image.rows, input_w = image.cols;
        float r = std::min(float(__height) / input_h, float(__width) / input_w ); // 取更小的缩小比例
        r = std::min(r, float(1)); // r要小于1，保证输入图片进网络是做缩小操作
        int new_unpad[] = {int(round(input_w * r)), int(round(input_h * r))};  
        std::cout << "resize: h:" << new_unpad[1] << ", w:" << new_unpad[0] << std::endl;

        cv::Mat im;
        cv::resize(image, im, cv::Size(new_unpad[0],new_unpad[1]), cv::INTER_LINEAR); // size(w,h)
        int dh = __height - new_unpad[1], dw = __width - new_unpad[0]; // 算填充  
        int top = dh/2, bottom = dh-(dh/2);
        int left = dw/2, right = dw-(dw/2);
        cv::copyMakeBorder(im, out, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));  // add border
        pad = {top, bottom, left, right};
    }
    else
    {
        cv::resize(image,out,cv::Size(__height,__width),cv::INTER_NEAREST);
        pad = {0, 0, 0, 0};
    }
    std::cout << "pad:   top:" << pad[0] << ", bottom:" << pad[1] << ", left:" << pad[2] << ", right:" << pad[3] << std::endl;
}

void Preprocess::normal(cv::Mat &image)
{
    cv::Mat img;
    image.convertTo(img, CV_32F);
    img = img / 255.0;
    img = img - __mean;
    std::vector<cv::Mat> bgrChannels(3);
    cv::split(img, bgrChannels);
    for(int i=0; i<3; i++)
    {
        bgrChannels[i].convertTo(bgrChannels[i], CV_32FC1, 1.0 / __std[i]);
    }
    cv::merge(bgrChannels, image);
}

void Preprocess::forward(cv::Mat image, cv::Mat &out, std::vector<int> &pad)
{
    cv::Mat image_rgb;
    cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
    resize(image_rgb, out, pad);
    normal(out);
}