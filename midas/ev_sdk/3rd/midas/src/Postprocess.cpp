#include <opencv2/opencv.hpp>
#include <numeric> 
#include <iostream>
#include "Postprocess.hpp"

Postprocess::Postprocess(std::vector<int> pad)
{
    sum = std::accumulate(pad.begin(), pad.end(), 0);
    top =  pad[0];
    bottom = pad[1];
    left  = pad[2];
    right = pad[3];
}

void Postprocess::forward(cv::Mat image, cv::Mat &out)
{
    cv::minMaxLoc(image, &minValue, &maxValue, &minLoc, &maxLoc);
    std::cout << "min_depth:" << minValue << ", max_depth:"<< maxValue << std::endl; 
    out = (image - minValue) / (maxValue-minValue)  * 255.0;
    if (sum > 0)
    {
        if (top + bottom == 0)
        {
            x1=left;
            y1=0;
            x2=image.cols-right;
            y2=image.rows;
        }
        if (left + right == 0)
        {
            x1=0;
            y1=top;
            x2=image.cols;
            y2=image.rows-(top+bottom);

        }
        cv::Rect rect(x1, y1, x2, y2); 
        out = out(rect);
    }
}
