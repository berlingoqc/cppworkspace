#include "headers.h"

extern "C" cudaError_t StartKernel_Object_Detection(uchar3 *pArrayA, uchar* pArrayR,int cols,int rows);

void GPUSobel(cv::Mat& img, cv::Mat& out);
