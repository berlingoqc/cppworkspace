#include "../../../include/capture.hpp"
#include "../../../include/video.hpp"
//#include "../../../include/file.hpp"


class RGBtoBlackVideo : public PixelVideoModifier {
	private:
		// Matrix pour detecter les objets une fois l'image convertit
		uchar** matrix;

		// Nbr de pixel de chaque couleur dominant
		int nbrRed, nbrYellow, nbrGreen;

		bool Between(Vec3b v,int lowerH, int higherH, int lowerV) {
			return v[0] >= lowerH && v[0] <= higherH && v[2] >= lowerV;
		}

		void SetTo(Vec3b *vec,int value) { vec[0]=value; vec[1]=value; vec[2]=value; }
	public:
		RGBtoBlackVideo() {
		}

	protected:
		void ModifierMat(Mat& img) {
			cvtColor(img,img,CV_RGB2HSV);
		}
		void Modifier(Vec3b * vec) {
			if(Between(*vec,80,143,50)) {
				SetTo(vec,0);
			} else if(Between(*vec,80,143,50)) {
				SetTo(vec,0);
			} else if(Between(*vec,80,143,50)) {
				SetTo(vec,0);
			} else {
				SetTo(vec,255);
			}
		}
};


int main(int argv,char ** argc)
{

	// AJOUTER un truc pour loader les settings de couleur depuis un fichier yaml
	RGBtoBlackVideo video;
	video.Start();
}