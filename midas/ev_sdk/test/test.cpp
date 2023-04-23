#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <time.h>
#include <onnxruntime_cxx_api.h>  
#include <numeric> 

#include "Preprocess.hpp"
#include "Postprocess.hpp"
#include "Midas.hpp"

// class Preprocess
// {
// public:
//     Preprocess(int height, int width, cv::Scalar mean, std::vector<float> std, bool keep_aspect_ratio);
//     ~Preprocess();    
//     void resize(cv::Mat image, cv::Mat &out, std::vector<int> &pad);
//     void normal(cv::Mat &image);
//     void forward(cv::Mat image, cv::Mat &out, std::vector<int> &pad);

// private:
//     int __height;
//     int __width;
//     cv::Scalar __mean;
//     std::vector<float> __std;
//     bool __keep_aspect_ratio;

// };


// Preprocess::Preprocess(int height, int width, cv::Scalar mean, std::vector<float> std, bool keep_aspect_ratio)
// {
//     __height = height;
//     __width =  width;
//     __mean = mean;
//     __std = std;
//     __keep_aspect_ratio = keep_aspect_ratio;
// }

// void Preprocess::resize(cv::Mat image, cv::Mat &out, std::vector<int> &pad)
// {
//     if (__keep_aspect_ratio)
//     {
//         float input_h = image.rows, input_w = image.cols;
//         float r = std::min(float(__height) / input_h, float(__width) / input_w ); // 取更小的缩小比例
//         r = std::min(r, float(1)); // r要小于1，保证输入图片进网络是做缩小操作
//         int new_unpad[] = {int(round(input_w * r)), int(round(input_h * r))};  
//         std::cout << "resize: h:" << new_unpad[1] << ", w:" << new_unpad[0] << std::endl;

//         cv::Mat im;
//         cv::resize(image, im, cv::Size(new_unpad[0],new_unpad[1]), cv::INTER_LINEAR); // size(w,h)
//         int dh = __height - new_unpad[1], dw = __width - new_unpad[0]; // 算填充  
//         int top = dh/2, bottom = dh-(dh/2);
//         int left = dw/2, right = dw-(dw/2);
//         cv::copyMakeBorder(im, out, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));  // add border
//         pad = {top, bottom, left, right};
//     }
//     else
//     {
//         cv::resize(image,out,cv::Size(__height,__width),cv::INTER_NEAREST);
//         pad = {0, 0, 0, 0};
//     }
//     std::cout << "pad:   top:" << pad[0] << ", bottom:" << pad[1] << ", left:" << pad[2] << ", right:" << pad[3] << std::endl;
// }

// void Preprocess::normal(cv::Mat &image)
// {
//     cv::Mat img;
//     image.convertTo(img, CV_32F);
//     img = img / 255.0;
//     img = img - __mean;
//     std::vector<cv::Mat> bgrChannels(3);
//     cv::split(img, bgrChannels);
//     for(int i=0; i<3; i++)
//     {
//         bgrChannels[i].convertTo(bgrChannels[i], CV_32FC1, 1.0 / __std[i]);
//     }
//     cv::merge(bgrChannels, image);
// }

// void Preprocess::forward(cv::Mat image, cv::Mat &out, std::vector<int> &pad)
// {
//     cv::Mat image_rgb;
//     cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
//     resize(image_rgb, out, pad);
//     normal(out);
// }



// class Postprocess
// {
// public:
//     Postprocess(std::vector<int> pad);
//     ~Postprocess();
//     void forward(cv::Mat image, cv::Mat &out);

// private:
//     double minValue, maxValue;
//     cv::Point minLoc, maxLoc;
//     cv::Mat out;

//     int top, bottom, left, right, sum;
//     int x1, y1, x2, y2;
// };

// Postprocess::Postprocess(std::vector<int> pad)
// {
//     sum = std::accumulate(pad.begin(), pad.end(), 0);
//     top =  pad[0];
//     bottom = pad[1];
//     left  = pad[2];
//     right = pad[3];
// }

// void Postprocess::forward(cv::Mat image, cv::Mat &out)
// {
//     cv::minMaxLoc(image, &minValue, &maxValue, &minLoc, &maxLoc);
//     std::cout << "min_depth:" << minValue << ", max_depth:"<< maxValue << std::endl; 
//     out = (image - minValue) / (maxValue-minValue)  * 255.0;
//     if (sum > 0)
//     {
//         if (top + bottom == 0)
//         {
//             x1=left;
//             y1=0;
//             x2=image.cols-right;
//             y2=image.rows;
//         }
//         if (left + right == 0)
//         {
//             x1=0;
//             y1=top;
//             x2=image.cols;
//             y2=image.rows-(top+bottom);

//         }
//         cv::Rect rect(x1, y1, x2, y2); 
//         out = out(rect);
//     }
// }


// class Midas
// {
// public:
//     Midas(std::string modelpath);
//     ~Midas();
//     cv::Mat forward(cv::Mat image);
//     std::vector<float> mat2vector(cv::Mat img);

// private:
//     std::string __modelpath;
//     int net_h;
//     int net_w;
//     int net_c;
//     cv::Mat pre; //推理结果
// };

// Midas::Midas(std::string modelpath)
// {
//     __modelpath = modelpath;
// }

// std::vector<float> Midas::mat2vector(cv::Mat img) 
// {
// 	img.convertTo(img, CV_32FC3);
// 	//将rgb数据分离为单通道
// 	std::vector<cv::Mat> mv;
// 	cv::split(img, mv);
// 	std::vector<float> R = mv[0].reshape(1, 1);
// 	std::vector<float> G = mv[1].reshape(1, 1);
// 	std::vector<float> B = mv[2].reshape(1, 1);
// 	//RGB数据合并
// 	std::vector<float> input_data;
// 	input_data.insert(input_data.end(), R.begin(), R.end());
// 	input_data.insert(input_data.end(), G.begin(), G.end());
// 	input_data.insert(input_data.end(), B.begin(), B.end());
// 	return input_data;
// }

// cv::Mat Midas::forward(cv::Mat image)
// {
//     Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "midas");
//     Ort::SessionOptions sessionOptions = Ort::SessionOptions();

//     // sessionOptions.SetIntraOpNumThreads(1); // 使用1个线程执行op,若想提升速度，增加线程数
//     // OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0); //GPU加速
//     sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);  //设置图优化类型，ORT_ENABLE_ALL: 启用所有可能的优化  
//     Ort::Session session(env, (const char*)__modelpath.c_str(), sessionOptions);

//     size_t num_input_nodes = session.GetInputCount(); //模型输入，一般输入只有图像的话input_nodes为1
//     size_t num_output_nodes = session.GetOutputCount(); // 如果是多输出网络，就会是对应输出的数目
//     printf("Number of inputs = %zu\n", num_input_nodes);
//     printf("Number of output = %zu\n", num_output_nodes);

//     auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();  // 获取输入维度
//     auto output_dims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape(); // 获取输出维度        
//     net_c = input_dims[1];
//     net_h = input_dims[2];
//     net_w = input_dims[3];
//     std::cout << "input_dims:" <<  input_dims[0] << " " << input_dims[1] << " " << net_h << " " << net_w << std::endl;
//     std::cout << "output_dims:" << output_dims[0] << " " << output_dims[1] << " " << output_dims[2] << std::endl;

//     if (image.rows != net_h || image.cols != net_w)
//     {
//         printf("image size is wrong!");
//     }
//     else
//     {
//         Ort::AllocatorWithDefaultOptions allocator;
//         const char* input_name = session.GetInputName(0, allocator);  //获取输入name              
//         const char* output_name = session.GetOutputName(0, allocator); 	//获取输出name
//         std::cout << "input_name:" << input_name << std::endl; 
//         std::cout << "output_name: " << output_name << std::endl;
//         std::vector<const char*> input_names = { input_name };
//         std::vector<const char*> output_names = { output_name};
//         std::vector<const char*> input_node_names = { input_name};
//         std::vector<const char*> output_node_names = { output_name};

//         auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
//         std::vector<float> __input_image = mat2vector(image);
//         // 准备输入tensor
//         Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
//                     __input_image.data(),  
//                     __input_image.size(), //net_h*net_w*3
//                     input_dims.data(), 
//                     input_dims.size()  
//                     );
//         // 开始推理
//         // startTime = clock();
//         auto output_tensor = session.Run(Ort::RunOptions{nullptr}, 
//                     input_node_names.data(), 
//                     &input_tensor, 
//                     input_names.size(), 
//                     output_node_names.data(), 
//                     output_node_names.size()
//                     );
//         // endTime = clock();

//         float* output = output_tensor[0].GetTensorMutableData<float>();  //输出是一个vector
//         pre = cv::Mat_<float>(net_h, net_w); //定义一个(net_h*net_w)的矩阵
//         for (int i = 0; i < pre.rows; i++)
//         {
//             for (int j = 0; j < pre.cols; j++) //矩阵列数循环
//             {
//                 pre.at<float>(i, j) = output[j];
//             }
//             output += net_w;
//         }
//     }
//     return pre;
// }



int main()
{
    int net_h = 256, net_w = 256;
    cv::Scalar mean = cv::Scalar(0.485, 0.456, 0.406);
    std::vector<float> std = {0.229, 0.224, 0.225};
    bool keep_aspect_ratio = true;

    cv::Mat src = cv::imread("./data/1.png");  //默认uchar 
    std::string modelpath = "./model/model-small.onnx";

    if (src.empty())
    {
        printf("faied to load image ...\n");
    }
    else
    {
        cv::Mat image;
        std::vector<int> pad;
        cv::Mat out;

        printf("preprocess...\n");
        auto *preprocess = new Preprocess(net_h, net_w, mean, std, keep_aspect_ratio);
        preprocess->forward(src, image, pad);
        
        printf("\ninference...\n");
        auto *midas = new Midas(modelpath);
        cv::Mat pre = midas->forward(image);

        printf("\npostprocess...\n");
        auto *postprocess = new Postprocess(pad);
        postprocess->forward(pre, out);

        cv::imwrite("out.jpg", out);
        printf("done!\n");
    }
    return 0;
}


