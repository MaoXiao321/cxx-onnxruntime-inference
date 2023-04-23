#ifndef PREPROCESS_HPP
#define PREPROCESS_HPP

#include <opencv2/opencv.hpp>

class Preprocess
{
public:
    Preprocess(int height, int width, cv::Scalar mean, std::vector<float> std, bool keep_aspect_ratio);
    ~Preprocess();    
    void resize(cv::Mat image, cv::Mat &out, std::vector<int> &pad);
    void normal(cv::Mat &image);
    void forward(cv::Mat image, cv::Mat &out, std::vector<int> &pad);

private:
    int __height;
    int __width;
    cv::Scalar __mean;
    std::vector<float> __std;
    bool __keep_aspect_ratio;

};

#endif // PREPROCESS_HPP