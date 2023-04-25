#include <opencv2/opencv.hpp>
#include <string>
#include "math.h"  //pow
#include <iostream>  //ifstream
#include <fstream>
#include <onnxruntime_cxx_api.h>  
#include <cuda_provider_factory.h>
#include "Yolov5.hpp"

Yolov5::Yolov5(std::string modelpath, std::string classesFile, float mThresh, float mHIERThresh, float mNms)
{
    __modelpath = modelpath;
    __mThresh = mThresh;
    __mHIERThresh = mHIERThresh;
    __mNms = mNms;
    std::cout << "anchor置信度阈值:" << __mThresh << std::endl;
    std::cout << "anchor最大类别阈值:" << __mHIERThresh << std::endl;
    std::cout << "iou阈值:" << __mNms << std::endl;

    std::ifstream ifs(classesFile.c_str());
	std::string line;
	while (std::getline(ifs, line)) class_names.push_back(line);
	num_class = class_names.size();
    std::cout << "class num:" << num_class << std::endl;
}

void Yolov5::resize(cv::Mat image, cv::Mat &out, std::vector<int> &pad)
{
    if (keep_aspect_ratio)
    {
        float input_h = image.rows, input_w = image.cols;
        float r = std::min(float(net_h) / input_h, float(net_w) / input_w ); // 取更小的缩小比例
        r = std::min(r, float(1)); // r要小于1，保证输入图片进网络是做缩小操作
        new_unpad = {int(round(input_w * r)), int(round(input_h * r))};  
        std::cout << "resize: h:" << new_unpad[1] << ", w:" << new_unpad[0] << std::endl;

        cv::Mat im;
        cv::resize(image, im, cv::Size(new_unpad[0],new_unpad[1]), cv::INTER_LINEAR); // size(w,h)
        int dh = net_h - new_unpad[1], dw = net_w - new_unpad[0]; // 算填充  
        int top = dh/2, bottom = dh-(dh/2);
        int left = dw/2, right = dw-(dw/2);
        cv::copyMakeBorder(im, out, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));  // add border
        pad = {top, bottom, left, right};
    }
    else
    {
        cv::resize(image,out,cv::Size(net_h,net_w),cv::INTER_NEAREST);
        pad = {0, 0, 0, 0};
    }
    std::cout << "pad:   top:" << pad[0] << ", bottom:" << pad[1] << ", left:" << pad[2] << ", right:" << pad[3] << std::endl;
}

void Yolov5::normal(cv::Mat &image)
{
    cv::Mat img;
    image.convertTo(img, CV_32F);
    img = img / 255.0;
    img = img - mean;
    std::vector<cv::Mat> bgrChannels(3);
    cv::split(img, bgrChannels);
    for(int i=0; i<3; i++)
    {
        bgrChannels[i].convertTo(bgrChannels[i], CV_32FC1, 1.0 / std[i]);
    }
    cv::merge(bgrChannels, image);
}

std::vector<float> Yolov5::mat2vector(cv::Mat img) 
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
    
    // //另一种写法
    // int row = img.rows;
	// int col = img.cols;
	// input_data.resize(row * col * img.channels());
	// for (int c = 0; c < 3; c++)
	// {
	// 	for (int i = 0; i < row; i++)
	// 	{
	// 		for (int j = 0; j < col; j++)
	// 		{
	// 			float pix = img.ptr<uchar>(i)[j * 3 + 2 - c]; //(i)[j * 3]是B通道，(i)[j * 3+1]是G通道，(i)[j * 3+2]是R通道
	// 			input_data[c * row * col + i * col + j] = pix / 255.0; //RGB并归一化
	// 		}
	// 	}
	// }
}

void Yolov5::nms(std::vector<Yolov5::BoxInfo>& input_boxes)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size()); //存预测框面积
    for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}
    std::vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < int(input_boxes.size()); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);

			float w = (std::max)(float(0), xx2 - xx1 + 1);
			float h = (std::max)(float(0), yy2 - yy1 + 1);
			float inter = w * h; //两个预测框的交集面积
			float iou = inter / (vArea[i] + vArea[j] - inter); //iou
			if (iou >= __mNms)
			{
				isSuppressed[j] = true;
			}
		}
	}
    int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}

void Yolov5::forward(cv::Mat &image, std::vector<Yolov5::Object> &result)
{
    // Preprocess
    resize(image, net_img, pad);
    cv::cvtColor(net_img, net_img, cv::COLOR_BGR2RGB);
    normal(net_img);

    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov5");
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();

    sessionOptions.SetIntraOpNumThreads(1); // 使用1个线程执行op,若想提升速度，增加线程数
    OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0); //GPU加速
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);  //设置图优化类型，ORT_ENABLE_ALL: 启用所有可能的优化  
    Ort::Session session(env, (const char*)__modelpath.c_str(), sessionOptions);

    // size_t num_input_nodes = session.GetInputCount(); //模型输入，一般输入只有图像的话input_nodes为1
    // size_t num_output_nodes = session.GetOutputCount(); // 如果是多输出网络，就会是对应输出的数目
    // printf("Number of inputs = %zu\n", num_input_nodes);
    // printf("Number of output = %zu\n", num_output_nodes);

    auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();  // 获取输入维度
    auto output_dims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape(); // 获取输出维度
    net_c = input_dims[1];
    // net_h = input_dims[2];
    // net_w = input_dims[3];
    nout = output_dims[2];
    std::cout << "input_dims:" <<  input_dims[0] << " " << input_dims[1] << " " << net_h << " " << net_w << std::endl;
    std::cout << "output_dims:" << output_dims[0] << " " << output_dims[1] << " " << output_dims[2] << std::endl;

    if (net_img.rows != net_h || net_img.cols != net_w)
    {
        printf("image size is wrong!");
    }
    else
    {
        Ort::AllocatorWithDefaultOptions allocator;
        const char* input_name = session.GetInputName(0, allocator);  //获取输入name              
        const char* output_name = session.GetOutputName(0, allocator); 	//获取输出name
        // std::cout << "input_name:" << input_name << std::endl; 
        // std::cout << "output_name: " << output_name << std::endl;
        std::vector<const char*> input_names = { input_name };
        std::vector<const char*> output_names = { output_name};
        std::vector<const char*> input_node_names = { input_name};
        std::vector<const char*> output_node_names = { output_name};

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        std::vector<float> input_image = mat2vector(net_img);
        // 准备输入tensor
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                    input_image.data(),  
                    input_image.size(), //net_h*net_w*3
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

        float* pdata = output_tensor[0].GetTensorMutableData<float>();  //输出是一个vector
        std::vector<Yolov5::BoxInfo> generate_boxes;
        // 遍历3个scale
        for(int n=0; n<num_stride; n++)
        {
            int stride = std::pow(2, n+3); //缩小倍数
            int num_grid_x = (int)ceil((net_w / stride));  //特征图大小
            int num_grid_y = (int)ceil((net_h / stride));
            // 遍历特征图上3个scale的anchor
            for(int q=0; q<num_stride; q++)
            {
                float anchor_w = anchors[n * 6 + q * 2]; //anchor大小
                float anchor_h = anchors[n * 6 + q * 2 + 1];
                //遍历特征图
                for (int i = 0; i < num_grid_y; i++)
                {
                    for (int j = 0; j < num_grid_x; j++)
                    {
                        float box_score = pdata[4]; //预测框的置信度
                        if (box_score > __mThresh) //寻找得分最大的类别
                        {
                            float max_class_socre = 0; //最大类别得分
                            int max_ind = 0; //最大类别索引
                            for (int k = 0; k < num_class; k++)
                            {
                                if (pdata[k+5] > max_class_socre)
                                {
                                    max_class_socre = pdata[k+5];
                                    max_ind = k;
                                }
                            }
                            max_class_socre *= box_score; //乘以置信度才是最大类别得分
                            if (max_class_socre > __mHIERThresh) 
                            {
                                float cx = (pdata[0] * 2.f - 0.5f + j) * stride;  //cx
                                float cy = (pdata[1] * 2.f - 0.5f + i) * stride;   //cy
                                float w = powf(pdata[2] * 2.f, 2.f) * anchor_w;   //w
                                float h = powf(pdata[3] * 2.f, 2.f) * anchor_h;  //h

                                int padw = pad[2], padh=pad[0];
                                float ratioh = (float)image.rows / new_unpad[1], ratiow = (float)image.cols / new_unpad[0]; 
                                float xmin = (cx - padw - 0.5 * w) * ratiow; 
                                float ymin = (cy - padh - 0.5 * h) * ratioh;
                                float xmax = (cx - padw + 0.5 * w) * ratiow;
                                float ymax = (cy - padh + 0.5 * h) * ratioh;

                                //记录该anchor
                                generate_boxes.push_back(BoxInfo{xmin, ymin, xmax, ymax, max_class_socre, max_ind});
                            }
                        }
                        pdata += nout; //移动指针
                    }
                }
            }
        }
        nms(generate_boxes);
        for (size_t i = 0; i < generate_boxes.size(); ++i)
        {
            int xmin = int(generate_boxes[i].x1);
            int ymin = int(generate_boxes[i].y1);
            int xmax = int(generate_boxes[i].x2);
            int ymax = int(generate_boxes[i].y2);
            cv::rectangle(image, cv::Point(xmin, ymin), cv::Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), cv::Scalar(0, 0, 255), 2);
            std::string label = cv::format("%.2f", generate_boxes[i].score);
            label = class_names[generate_boxes[i].label] + ":" + label;
            result.push_back({ generate_boxes[i].score, class_names[generate_boxes[i].label], cv::Rect(xmin, ymin, xmax- xmin, ymax- ymin)});
            cv::putText(image, label, cv::Point(xmin, ymin - 5), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
        }
    }
}
