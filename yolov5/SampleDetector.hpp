#ifndef JI_SAMPLEDETECTOR_HPP
#define JI_SAMPLEDETECTOR_HPP
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>

using namespace cv;
using namespace std;
using namespace Ort;


//都是自己的东西，对照着cpp写

class SampleDetector {

public:
    typedef struct {
        float prob;
        std::string name;
        cv::Rect rect;
    } Object;
    typedef struct {
        // 算法配置可选的配置参数
        double nms;
        double thresh;
        double hierThresh;
    } ALGO_CONFIG_TYPE;
    typedef struct BoxInfo
    {
        float x1;
        float y1;
        float x2;
        float y2;
        float score;
        int label;
    } BoxInfo;
    SampleDetector(double thresh, double nms, double hierThresh);
    //int init(const char* namesFile,const char* weightsFile);
    int init(string namesFile, string weightsFile);
    void unInit();
    int detect(const Mat& frame, vector<Object>& result);

public:
    static const int ERROR_BASE = 0x0100;
    static const int ERROR_INVALID_INPUT = 0x0101;
    static const int ERROR_INVALID_INIT_ARGS = 0x0102;
    static const int PROCESS_OK = 0x1001;
    static const int INIT_OK = 0x1002;

private:
    vector<string> class_names;
    int num_class;
    int inpWidth;
    int inpHeight;
    int nout;
    int num_proposal;
    int num_stride;
    float* anchors;

    double mThresh;
    double mHIERThresh; //置信度*类别得分
    double mNms;
    const bool keep_ratio = true;
    vector<float> input_image_;
    Mat resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left);
    void normalize_(Mat img);
    void nms(vector<BoxInfo>& input_boxes);

    //onnxruntime
    Env env = Env(ORT_LOGGING_LEVEL_ERROR, "yolov5-6.1");
    Ort::Session* ort_session = nullptr;
    SessionOptions sessionOptions = SessionOptions();
    vector<char*> input_names;
    vector<char*> output_names;
    vector<vector<int64_t>> input_node_dims; // >=1 outputs
    vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

#endif