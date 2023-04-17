#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>
#include "SampleDetector.hpp"

using namespace std;
using namespace Ort;

const float anchors_640[3][6] = { {10.0,  13.0, 16.0,  30.0,  33.0,  23.0},
								 {30.0,  61.0, 62.0,  45.0,  59.0,  119.0},
								 {116.0, 90.0, 156.0, 198.0, 373.0, 326.0} };

const float anchors_1280[4][6] = { {19, 27, 44, 40, 38, 94},
								{96, 68, 86, 152, 180, 137},
								{140, 301, 303, 264, 238, 542},
								{436, 615, 739, 380, 925, 792} };

SampleDetector::SampleDetector(double thresh, double nms, double hierThresh) :
	mNms(nms), mThresh(thresh), mHIERThresh(hierThresh) {
	cout << "Current config: nms:" << mNms << ", thresh:" << mThresh
		<< ", HIERThresh:" << mHIERThresh;
}


int endsWith(string s, string sub) {
	//字符串从后往前搜索指定内容，返回索引值，如果没有就返回-1。当索引值等于s.length() - sub.length()时，说明以sub结尾
	return s.rfind(sub) == (s.length() - sub.length()) ? 1 : 0;
}

int SampleDetector::init(string classesFile, string model_path) {
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions); //ort_session用于预测
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	this->nout = output_node_dims[0][2];
	this->num_proposal = output_node_dims[0][1];

	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	this->num_class = class_names.size();

	if (endsWith(model_path, "6.onnx"))
	{
		anchors = (float*)anchors_1280;
		this->num_stride = 4;
	}
	else
	{
		anchors = (float*)anchors_640;
		this->num_stride = 3;
	}
	return SampleDetector::INIT_OK;
}

void SampleDetector::unInit() {
}

Mat SampleDetector::resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left)
{
	//指针是一个变量，用来存地址。指针前再加个*是解引用，拿出这个地址存的值
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = inpHeight; //newh是指针，指针前加*拿出该位置的值，这里是将该位置的值赋为inpHeight
	*neww = inpWidth;

	Mat dstimg;
	if (keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw; //高宽比
		//调整图片大小
		if (hw_scale > 1) {
			*newh = inpHeight;
			*neww = int(inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, inpWidth - *neww - *left, BORDER_CONSTANT, 114);
		}
		else {
			*newh = (int)inpHeight * hw_scale;
			*neww = inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 114);
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void SampleDetector::normalize_(Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
				this->input_image_[c * row * col + i * col + j] = pix / 255.0;

			}
		}
	}
}

void SampleDetector::nms(vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
	vector<float> vArea(input_boxes.size());
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}

	vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < int(input_boxes.size()); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = (max)(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = (max)(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = (min)(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = (min)(input_boxes[i].y2, input_boxes[j].y2);

			float w = (max)(float(0), xx2 - xx1 + 1);
			float h = (max)(float(0), yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (vArea[i] + vArea[j] - inter);

			if (ovr >= this->mNms)
			{
				isSuppressed[j] = true;
			}
		}
	}
	// return post_nms;
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}

int SampleDetector::detect(const Mat& frame, vector<Object>& result)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	/////generate proposals
	vector<BoxInfo> generate_boxes;
	float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
	int n = 0, q = 0, i = 0, j = 0, row_ind = 0, k = 0; ///xmin,ymin,xamx,ymax,box_score, class_score
	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
	for (n = 0; n < this->num_stride; n++)   ///特征图尺度
	{
		const float stride = pow(2, n + 3);
		int num_grid_x = (int)ceil((this->inpWidth / stride));
		int num_grid_y = (int)ceil((this->inpHeight / stride));
		for (q = 0; q < 3; q++)    ///anchor
		{
			const float anchor_w = this->anchors[n * 6 + q * 2];
			const float anchor_h = this->anchors[n * 6 + q * 2 + 1];
			for (i = 0; i < num_grid_y; i++)
			{
				for (j = 0; j < num_grid_x; j++)
				{
					float box_score = pdata[4];
					if (box_score > this->mThresh)
					{
						int max_ind = 0;
						float max_class_socre = 0;
						for (k = 0; k < num_class; k++)
						{
							if (pdata[k + 5] > max_class_socre)
							{
								max_class_socre = pdata[k + 5];
								max_ind = k;
							}
						}
						max_class_socre *= box_score;
						if (max_class_socre > this->mHIERThresh)
						{
							float cx = (pdata[0] * 2.f - 0.5f + j) * stride;  ///cx
							float cy = (pdata[1] * 2.f - 0.5f + i) * stride;   ///cy
							float w = powf(pdata[2] * 2.f, 2.f) * anchor_w;   ///w
							float h = powf(pdata[3] * 2.f, 2.f) * anchor_h;  ///h

							float xmin = (cx - padw - 0.5 * w) * ratiow; 
							float ymin = (cy - padh - 0.5 * h) * ratioh;
							float xmax = (cx - padw + 0.5 * w) * ratiow;
							float ymax = (cy - padh + 0.5 * h) * ratioh;

							generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, max_class_socre, max_ind });
						}
					}
					row_ind++;
					pdata += nout;
				}
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	nms(generate_boxes);
	for (size_t i = 0; i < generate_boxes.size(); ++i)
	{
		int xmin = int(generate_boxes[i].x1);
		int ymin = int(generate_boxes[i].y1);
		int xmax = int(generate_boxes[i].x2);
		int ymax = int(generate_boxes[i].y2);
		rectangle(frame, Point(xmin, ymin), Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), Scalar(0, 0, 255), 2);
		string label = format("%.2f", generate_boxes[i].score);
		label = this->class_names[generate_boxes[i].label] + ":" + label;
		result.push_back({ generate_boxes[i].score, class_names[generate_boxes[i].label], Rect(xmin, ymin, xmax- xmin, ymax- ymin)});
		putText(frame, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	}
	return SampleDetector::PROCESS_OK;
}
