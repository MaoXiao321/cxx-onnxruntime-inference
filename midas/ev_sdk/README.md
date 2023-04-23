## 代码结构
midas的前处理、后处理等打包成libmidas.so，test.cpp中调用该动态库。

## onnxruntime
3rd/onnxruntime需要根据操作系统+CPU/GPU+cuda版本来决定使用的onnxruntime版本。这里的onnxruntime是1.6-windows。若当前环境为 Ubuntu18.04 + GPU + cuda11，可下载onnxruntime-linux-x64-gpu-1.7.0.tgz，下载地址https://github.com/microsoft/onnxruntime/releases/tag/v1.7.0。

3rd/midas/src/Midas.cpp加上以下两句：
```
#include <cuda_provider_factory.h>
OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0); //GPU加速
```
