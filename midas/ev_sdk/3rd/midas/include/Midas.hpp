#ifndef MIDAS_HPP  
#define MIDAS_HPP

#include <opencv2/opencv.hpp>
#include <string>

class Midas
{
public:
    Midas(std::string modelpath);
    ~Midas();
    cv::Mat forward(cv::Mat image);
    std::vector<float> mat2vector(cv::Mat img);

private:
    std::string __modelpath;
    int net_h;
    int net_w;
    int net_c;
    cv::Mat pre; //推理结果
};

#endif