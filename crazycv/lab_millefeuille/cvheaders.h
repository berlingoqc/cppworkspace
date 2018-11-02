#ifndef CVHEADERS_H
#define CVHEADERS_H


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/ocl.hpp>
#ifdef HAVE_OPENCV_CONTRIB
	#include <opencv2/tracking.hpp>
#endif

#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

#include <iostream>
#include <string>



struct thresh_range {
	int H_MIN;
	int H_MAX;
	int S_MIN;
	int S_MAX;
	int V_MIN;
	int V_MAX;

	int T_MIN;
	int T_MAX;
};


#endif // CVHEADERS_H
