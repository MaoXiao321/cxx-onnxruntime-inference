#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include "Midas.hpp"
#include <onnxruntime_cxx_api.h> 

Midas::Midas(std::string modelpath)
{
    __modelpath = modelpath;
}

std::vector<float> Midas::mat2vector(cv::Mat img) 
{
	img.convertTo(img, CV_32FC3);
	//将rgb数据分离为单通道
	std::vector<cv::Mat> mv;
	cv::split(img, mv);
	std::vector<float> R = mv[0].reshape(1, 1);
	std::vector<float> G = mv[1].reshape(1, 1);
	std::vector<float> B = mv[2].reshape(1, 1);
	//RGB数据合并
	std::vector<float> input_data;
	input_data.insert(input_data.end(), R.begin(), R.end());
	input_data.insert(input_data.end(), G.begin(), G.end());
	input_data.insert(input_data.end(), B.begin(), B.end());
	return input_data;
}

cv::Mat Midas::forward(cv::Mat image)
{
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "midas");
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();

    // sessionOptions.SetIntraOpNumThreads(1); // 使用1个线程执行op,若想提升速度，增加线程数
    // OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0); //GPU加速
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);  //设置图优化类型，ORT_ENABLE_ALL: 启用所有可能的优化  
    Ort::Session session(env, (const char*)__modelpath.c_str(), sessionOptions);

    size_t num_input_nodes = session.GetInputCount(); //模型输入，一般输入只有图像的话input_nodes为1
    size_t num_output_nodes = session.GetOutputCount(); // 如果是多输出网络，就会是对应输出的数目
    printf("Number of inputs = %zu\n", num_input_nodes);
    printf("Number of output = %zu\n", num_output_nodes);

    auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();  // 获取输入维度
    auto output_dims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape(); // 获取输出维度        
    net_c = input_dims[1];
    net_h = input_dims[2];
    net_w = input_dims[3];
    std::cout << "input_dims:" <<  input_dims[0] << " " << input_dims[1] << " " << net_h << " " << net_w << std::endl;
    std::cout << "output_dims:" << output_dims[0] << " " << output_dims[1] << " " << output_dims[2] << std::endl;

    if (image.rows != net_h || image.cols != net_w)
    {
        printf("image size is wrong!");
    }
    else
    {
        Ort::AllocatorWithDefaultOptions allocator;
        const char* input_name = session.GetInputName(0, allocator);  //获取输入name              
        const char* output_name = session.GetOutputName(0, allocator); 	//获取输出name
        std::cout << "input_name:" << input_name << std::endl; 
        std::cout << "output_name: " << output_name << std::endl;
        std::vector<const char*> input_names = { input_name };
        std::vector<const char*> output_names = { output_name};
        std::vector<const char*> input_node_names = { input_name};
        std::vector<const char*> output_node_names = { output_name};

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        std::vector<float> __input_image = mat2vector(image);
        // 准备输入tensor
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                    __input_image.data(),  
                    __input_image.size(), //net_h*net_w*3
                    input_dims.data(), 
                    input_dims.size()  
                    );
        // 开始推理
        // startTime = clock();
        auto output_tensor = session.Run(Ort::RunOptions{nullptr}, 
                    input_node_names.data(), 
                    &input_tensor, 
                    input_names.size(), 
                    output_node_names.data(), 
                    output_node_names.size()
                    );
        // endTime = clock();

        float* output = output_tensor[0].GetTensorMutableData<float>();  //输出是一个vector
        pre = cv::Mat_<float>(net_h, net_w); //定义一个(net_h*net_w)的矩阵
        for (int i = 0; i < pre.rows; i++)
        {
            for (int j = 0; j < pre.cols; j++) //矩阵列数循环
            {
                pre.at<float>(i, j) = output[j];
            }
            output += net_w;
        }
    }
    return pre;
}