#ifndef POSTPROCESS_HPP
#define POSTPROCESS_HPP

#include <opencv2/opencv.hpp>
class Postprocess
{
public:
    Postprocess(std::vector<int> pad);
    ~Postprocess();
    void forward(cv::Mat image, cv::Mat &out);

private:
    double minValue, maxValue;
    cv::Point minLoc, maxLoc;
    cv::Mat out;

    int top, bottom, left, right, sum;
    int x1, y1, x2, y2;
};

#endif //POSTPROCESS_HPP
