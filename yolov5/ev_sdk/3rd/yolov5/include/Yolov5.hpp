#ifndef YOLOV5_HPP
#define YOLOV5_HPP

#include <opencv2/opencv.hpp>
#include <string>


class Yolov5
{
public:
    typedef struct BoxInfo
    {
        float x1;
        float y1;
        float x2;
        float y2;
        float score;
        int label;
    } BoxInfo;
    typedef struct {
        float prob;
        std::string name;
        cv::Rect rect;
    } Object;

    Yolov5(std::string modelpath, std::string classesFile, float mThresh, float mHIERThresh, float mNms);
    ~Yolov5();

    //Preprocess
    void resize(cv::Mat image, cv::Mat &out, std::vector<int> &pad);
    void normal(cv::Mat &image);
    
    //inference
    std::vector<float> mat2vector(cv::Mat img);
    void nms(std::vector<Yolov5::BoxInfo>& input_boxes);
    void forward(cv::Mat &image, std::vector<Yolov5::Object> &result);

private:
    //Preprocess
    bool keep_aspect_ratio=true;
    cv::Scalar mean= cv::Scalar(0, 0, 0);
    std::vector<float> std = {1.0, 1.0, 1.0};
    std::vector<int> pad;
    std::vector<int> new_unpad;
    int net_h = 640;
    int net_w = 640;
    cv::Mat net_img; // 前处理后的图片

    std::string __modelpath;
    float __mThresh; //预测框的置信度阈值
    float __mHIERThresh; //最大类别得分阈值
    float __mNms; //iou阈值
    std::vector<std::string> class_names;
    int num_class; //类别数
    int net_c; //通道数
    int nout; //一个anchor有nout个推理结果(x,y,w,h,score,各类别得分)
    cv::Mat pre; //推理结果

    int num_stride = 3;
    const float anchors_640[3][6] = { {10.0,  13.0, 16.0,  30.0,  33.0,  23.0},
								 {30.0,  61.0, 62.0,  45.0,  59.0,  119.0},
								 {116.0, 90.0, 156.0, 198.0, 373.0, 326.0} };
    float* anchors = (float*)anchors_640;
    
};

#endif