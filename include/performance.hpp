
#include <opencv2/core/core.hpp>

// MeasureTime est une classe pour mesurer la durer en temps d'une execution en seconde
class MeasureTime {
    
    double startedtime;
    double stopedtime;

    public:
        MeasureTime() {}
        double Stop() {
            stopedtime = (double)cv::getTickCount();
            return GetElapsedTime();
        }

        double GetElapsedTime() {
            if(stopedtime == 0)
                return 0;
            return (startedtime-stopedtime)/cv::getTickFrequency();
        }

        void Start() {
            startedtime = (double)cv::getTickCount();
        }


};