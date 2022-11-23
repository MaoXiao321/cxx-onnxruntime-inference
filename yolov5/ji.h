#ifndef JI_H //#ifndef..define..endif是为了防止重编译
#define JI_H
// 没有要改的

#define EV_SDK_VERSION  "3.0.3"

// 函数返回值定义
#define JISDK_RET_SUCCEED               (0)             // 成功
#define JISDK_RET_FAILED                (-1)            // 失败
#define JISDK_RET_UNUSED                (-2)            // 未实现
#define JISDK_RET_INVALIDPARAMS         (-3)            // 参数错误
#define JISDK_RET_OFFLINE               (-9)            // 联网校验时离线状态
#define JISDK_RET_OVERMAXQPS            (-99)           // 超过最大请求量
#define JISDK_RET_UNAUTHORIZED          (-999)          // 未授权

// ji_create_predictor可选输入参数定义
#define JISDK_PREDICTOR_DEFAULT         (0)             // 默认
#define JISDK_PREDICTOR_SEQUENTIAL      (1)             // 连续的，即带状态的
#define JISDK_PREDICTOR_NONSEQUENTIAL   (2)             // 非连续的，即不带状态的

// JI_EVENT.code值定义
#define JISDK_CODE_ALARM                (0)             // 报警
#define JISDK_CODE_NORMAL               (1)             // 正常
#define JISDK_CODE_FAILED               (-1)            // 失败

typedef struct {
	int rows;           // cv::Mat::rows
	int cols;           // cv::Mat::cols
	int type;           // cv::Mat::type()
	void* data;         // cv::Mat::data
	int step;           // cv::Mat::step
} JI_CV_FRAME;

typedef struct {
	int code;           // 详见"JI_EVENT.code值定义"
	const char* json;   // 算法输出结果，json格式的字符串
} JI_EVENT;

void* ji_create_predictor(int pdtype);

int ji_calc_frame(void* predictor, const JI_CV_FRAME* inFrame, const char* args,
	JI_CV_FRAME* outFrame, JI_EVENT* event);
#endif
