## onnxruntime
3rd/onnxruntime需要根据操作系统+CPU/GPU+cuda版本来决定使用的onnxruntime版本。若当前环境为 Ubuntu18.04 + GPU + cuda11，可下载onnxruntime-linux-x64-gpu-1.7.0.tgz，下载地址https://github.com/microsoft/onnxruntime/releases/tag/v1.7.0。
将压缩包中的include和lib放到onnxruntime下。

## model
从https://github.com/ultralytics/yolov5下载权重文件，用export.py将pt转成onnx

## 代码结构
将前处理、后处理、推理等打包成3rd/yolov5/lib/libyolov5.so
```
cd 3rd/yolov5
rm -rf build && mkdir -p build && cd build && cmake .. && make install && cd ..
```
test.cpp中调用该动态库，生成可执行文件bin/test-ji-api
```
bash run.sh
```
