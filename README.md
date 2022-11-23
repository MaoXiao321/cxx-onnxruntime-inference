## yolov5
C++推理yolov5.onnx，pt文件转onnx采用https://github.com/ultralytics/yolov5 中的export.py。具体使用写在：https://github.com/MaoXiao321/pytorch-yolov5


## 代码结构

test.cpp中main函数为主入口：

第一步：创建检测器
predictor = ji_create_predictor(JISDK_PREDICTOR_DEFAULT); 

第二步：ji_calc_frame()
main函数调用test_for_ji_calc_frame()，test_for_ji_calc_frame调用ji_calc_frame()
iRet = ji_calc_frame(predictor, &inframe, EMPTY_EQ_NULL(strArgs), &outframe, &event); 

第三步：processMat()
ji_calc_frame()调用processMat()对输入数据进行预处理，并返回预测结果。
int processRet = processMat(detector, inMat, args, outputFrame, *event); 

第四步：detector
detector的定义及相关操作写在SampleDetector.cpp中。
