#include "../../../include/capture.hpp"
#include "../../../include/video.hpp"
//#include "../../../include/file.hpp"


class RGBtoBlackVideo : public PixelVideoModifier {
	private:
		// Valeur de d'interval pour mes différents couleurs a detecter
		cv::Vec3b lower_green, higher_green;
		cv::Vec3b lower_red, higher_red;
		cv::Vec3b lower_yellow, higher_yellow;

		// Matrix pour detecter les objets une fois l'image convertit
		uchar** matrix;

		// Nbr de pixel de chaque couleur dominant
		int nbrRed, nbrYellow, nbrGreen;

		bool Between(Vec3b v, Vec3b top, Vec3b bot) {
			return v[0] >= bot[0] && v[0] <= top[0] && v[1] >= bot[1] && v[1] <= top[1] && v[2] >= bot[2] && v[2] <= top[2];
		}

	protected:
		void Modifier(Vec3b * vec) {
			if(Between(vec,lower_blue,higher_blue)) {
				vec = Vec3b(0,0,0);
			} else if(Between(vec,lower_red,higher_red)) {
				vec = Vec3b(0,0,0);
			} else if(Between(vec,lower_yellow,higher_yellow)) {
				vec = Vec3b(0,0,0);
			} else {
				vec = Vec3b(255,255,255);
			}
		}
};


int main(int argv,char ** argc)
{

	// AJOUTER un truc pour loader les settings de couleur depuis un fichier yaml
	RGBtoBlackVideo video;
	video.Start();
}