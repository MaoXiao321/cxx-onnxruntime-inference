#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include "text_det.h"
#include "text_angle_cls.h"
#include "text_rec.h"

using namespace cv;
using namespace std;
using namespace Ort;


int main()
{
	TextDetector detect_model("model/brass_det_1214.onnx");
	TextClassifier angle_model("model/ch_ppocr_mobile_v2.0_cls_infer.onnx");
	TextRecognizer rec_model("model/ch_PP-OCRv3_rec_infer.onnx", "model/ppocr_keys_v1.txt");

	string imgpath = "data/brass.jpg";
	Mat srcimg = imread(imgpath);
	///cv::rotate(srcimg, srcimg, 1);

	vector< vector<Point2f> > results = detect_model.detect(srcimg);
	cout << "text box num:" << results.size() << endl;
	vector<string> labels;

	for (size_t i = 0; i < results.size(); i++)
	{
		Mat textimg = detect_model.get_rotate_crop_image(srcimg, results[i]);	
		if (angle_model.predict(textimg) == 1)
		{
			cv::rotate(textimg, textimg, 1);
		}

		string text = rec_model.predict_text(textimg);
		// cout <<  text << endl;
		labels.push_back(text);
	}
	detect_model.draw_pred(srcimg, results, labels);
	imwrite("result.jpg", srcimg);
	return 0;
}