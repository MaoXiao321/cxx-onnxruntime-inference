#ifndef JI_H //#ifndef..define..endif��Ϊ�˷�ֹ�ر���
#define JI_H
// û��Ҫ�ĵ�

#define EV_SDK_VERSION  "3.0.3"

// ��������ֵ����
#define JISDK_RET_SUCCEED               (0)             // �ɹ�
#define JISDK_RET_FAILED                (-1)            // ʧ��
#define JISDK_RET_UNUSED                (-2)            // δʵ��
#define JISDK_RET_INVALIDPARAMS         (-3)            // ��������
#define JISDK_RET_OFFLINE               (-9)            // ����У��ʱ����״̬
#define JISDK_RET_OVERMAXQPS            (-99)           // �������������
#define JISDK_RET_UNAUTHORIZED          (-999)          // δ��Ȩ

// ji_create_predictor��ѡ�����������
#define JISDK_PREDICTOR_DEFAULT         (0)             // Ĭ��
#define JISDK_PREDICTOR_SEQUENTIAL      (1)             // �����ģ�����״̬��
#define JISDK_PREDICTOR_NONSEQUENTIAL   (2)             // �������ģ�������״̬��

// JI_EVENT.codeֵ����
#define JISDK_CODE_ALARM                (0)             // ����
#define JISDK_CODE_NORMAL               (1)             // ����
#define JISDK_CODE_FAILED               (-1)            // ʧ��

typedef struct {
	int rows;           // cv::Mat::rows
	int cols;           // cv::Mat::cols
	int type;           // cv::Mat::type()
	void* data;         // cv::Mat::data
	int step;           // cv::Mat::step
} JI_CV_FRAME;

typedef struct {
	int code;           // ���"JI_EVENT.codeֵ����"
	const char* json;   // �㷨��������json��ʽ���ַ���
} JI_EVENT;

void* ji_create_predictor(int pdtype);

int ji_calc_frame(void* predictor, const JI_CV_FRAME* inFrame, const char* args,
	JI_CV_FRAME* outFrame, JI_EVENT* event);
#endif
