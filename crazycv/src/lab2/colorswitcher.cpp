#include "../../../include/capture.hpp"
#include "../../../include/video.hpp"
//#include "../../../include/file.hpp"


class RGBtoBlackVideo : public PixelVideoModifier {
	Vec3b Modifier(Vec3b vec) {
        int r = vec[2];
        int g = vec[1];
        int b = vec[0];
        //printf("X : %d Y : %d R : %d G: %d B: %d\n",x,y,r,v,b);
	    // Detection du rouge
	    if (r > 140 && r <= 255 && g < 100 & b < 100) {
           	vec = Vec3b(0,0,0);
    	} else if ( r > 240 && g > 240 && b < 130) {
			vec = Vec3b(0,0,0);
		} else if ( g > 140 && r < 100 & b < 100 ) {
			vec = Vec3b(0,0,0);
		}
		return vec;
}
};

class ContBrighVideo : public PixelVideoModifier {
	double 	alpha;
	int 	beta;

	public:
		ContBrighVideo(double a,int b) {
			alpha = a;
			beta = b;
		}

	protected:

		virtual Vec3b Modifier(Vec3b vec) {
			for (int c = 0;c < 3; c++) {
				vec[c] = saturate_cast<uchar>(alpha*vec[c]+beta);
			}
			return vec;
		}

		virtual bool HandleKey(int k) {
			switch(k) {
				// descend alpha
				case 'a':
					if(alpha > 1.0) alpha -= 0.10;
				break;
				// monte alpha
				case 'd':
					if(alpha < 3.0) alpha += 0.10;
				break;
				// monte beta
				case 'w':
					if(beta > 0) beta -= 10;
				break;
				// descend beta
				case 's':
					if(beta < 100) beta += 10;
				break;
				case 'q':
					return true;
				break;
			}
			return false;
		}
};

int main(int argv,char ** argc)
{


	printf(getBuildInformation().c_str());

	ContBrighVideo video(0.5,50);
	video.Start();
	/*
	RGBtoBlackVideo video;
	video.Start();
	*/
}