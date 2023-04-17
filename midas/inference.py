import numpy as np
import onnxruntime
import cv2

class Preprocess():
    """preprocess.
    """

    def __init__(self,height, width, mean, std, keep_aspect_ratio=False):
        self.__height = height
        self.__width = width
        self.__mean = mean
        self.__std = std
        self.__keep_aspect_ratio = keep_aspect_ratio
    
    def resize(self, image):
        if self.__keep_aspect_ratio:
            input_h, input_w  = image.shape[0], image.shape[1]
            r = min(self.__height / input_h, self.__width / input_w ) # 取更小的缩小比例
            r = min(r, 1.0) # r要小于1，保证输入图片进网络是做缩小操作
            new_unpad = int(round(input_w * r)), int(round(input_h * r)) # 缩放后大小(w，h)
            if (self.__height, self.__width) != new_unpad:  
                im = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)      
            dh, dw = self.__height - new_unpad[1], self.__width - new_unpad[0]  # 算填充   
            top,bottom = dh//2, dh-(dh//2)
            left,right = dw//2, dw -(dw//2)
            color=(0, 0, 0)
            out = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
            pad = (top, bottom, left, right)
            cv2.imwrite("pad.jpg", out)
        else:
            out = cv2.resize(image, (self.__height, self.__width), interpolation=cv2.INTER_NEAREST)
            pad = None
            cv2.imwrite("pad.jpg", out)
        return out,pad

    def normal(self, image):
        image = (image - self.__mean) / self.__std
        # for i in range(3):
        #     x[i] = (x[i] - mean[i]) / std[i]   
        return image
    
    def forward(self, image):
        x,pad = self.resize(image)
        x = np.array(x / 255)
        x = self.normal(x)
        x = x.transpose(-1, 0, 1) # (c, h, w) 
        x = np.reshape(x, [1, x.shape[0], x.shape[1], x.shape[2]]) # (b, c, h, w)
        x = np.float32(x)
        return x,pad

class Postprocess():
    """postprocess.
    """
    def __init__(self) -> None:
        pass
    
    def forward(self, image, pad):
        depth_max = np.max(image)
        depth_min = np.min(image)
        out = (image - depth_min) / (depth_max - depth_min)
        out = out * 255
        if pad:
            top, bottom, left, right = pad[0], pad[1], pad[2], pad[3]
            if top + bottom == 0:
                x1,y1 = left,0
                x2,y2 = image.shape[1]-right,image.shape[0]
            if left + right == 0:
                x1,y1 = 0,top
                x2,y2 = image.shape[1],image.shape[0]-(top+bottom)
            out = out[y1:y2, x1:x2]
        return out

class ONNX():
    def __init__(self, path='model-f6b98070.onnx'):
        self.__path = path
    
    def forward(self, x):
        ort_session = onnxruntime.InferenceSession(self.__path)
        ort_inputs = {ort_session.get_inputs()[0].name: x}
        ort_outs = ort_session.run(None, ort_inputs)
        depth = ort_outs[0][0]
        return depth


if __name__ == '__main__':  
    model = "model-f6b98070.onnx"
    if model == "model-f6b98070.onnx":
        net_h, net_w = 384, 384
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    if model == "model-small.onnx":
        net_h, net_w = 256, 256
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    original_image = cv2.imread("input/1.png")    

    # preprocess
    preprocess = Preprocess(net_h,net_w,mean,std,keep_aspect_ratio=True)
    x,pad = preprocess.forward(original_image)

    # compute ONNX Runtime output prediction
    onnx_model = ONNX(f'model/{model}')
    prediction = onnx_model.forward(x)

    # postprocess
    postprocess = Postprocess()
    out = postprocess.forward(prediction, pad)

    cv2.imwrite("output/depth.jpg", out)