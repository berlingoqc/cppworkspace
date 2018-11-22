#ifndef __ENGINE_H__
#define __ENGINE_H__


#include "header.h"



namespace ENGINE
{
#define PI 3.14159265

	typedef float Mat4x4[4][4];

	const Mat4x4 MatriceTransformation{
		{ 1, 0, 0, 0},
		{ 0, 1, 0, 0},
		{ 0, 0, 1, 0},
		{ 0, 0 ,0, 1}
	};

	// Constante de l'écran
	const int DEFAULT_SCREEN_WIDTH = 800;
	const int DEFAULT_SCREEN_HEIGHT = 600;
	const int SCREEN_FPS = 60;


	const float Maxndc = 1.0f;
	const float Minndc = -1.0f;


	struct Vecf {
		float x, y;
		Vecf() : x(0), y(0) {}
		Vecf(float a, float b) : x(a), y(b) {}

		Vecf operator+(const Vecf& p) {
			return Vecf(x + p.x, y + p.y);
		}
		Vecf operator-(const Vecf& p) {
			return Vecf(x - p.x, y - p.y);
		}


		void add(const Vecf& p) {
			x += p.x;
			y += p.y;
		}
		void sub(const Vecf& p) {
			x -= p.x;
			y -= p.y;
		}
		void scale(float n) {
			x = x * n;
			y = y * n;
		}

		float magnitude() {
			return sqrt(x*x + y * y);
		}

		void normalize() {
			float mag = magnitude();
			if (mag != 0) {
				scale(1 / mag);
			}
		}

		Vecf rotate(float degrees) {
			double theta = (degrees * PI / 180.0f);
			double cosVal = cos(theta);
			double sinVal = sin(theta);
			double newX = x * cosVal - y * sinVal;
			double newY = x * sinVal + y * cosVal;
			return Vecf(newX, newY);
		}
	};


	template<typename T>
	struct Position {
		T x;
		T y;
	};



	template<typename T>
	struct MyLine {
		Position<T> p1;
		Position<T> p2;

	};

	struct VecLine {
		Position<float> p;
		Vecf v;
	};


	template<typename T>
	struct TrigoInfo {
		T A; // Longeur de coté adjacent
		T H; // Longeur de l'hypothenuse
		T O; // Longeur du coté opposé

		T Angle; // Angle du triangle rectangle
	};



	template<typename T>
	struct RGBColor {
		T R;
		T G;
		T B;
	};

	// enum screenpositions contient les positions possible pour placer rapidement notre fenetre
	enum screenpositions { topleft, topright, center, bottomleft, bottomright };

	// APPINFO est une structure qui contient les informations relatives au démarre d'une nouvelle application OpenGL
	struct APPINFO {
		char title[128];
		int windowWidth;
		int windowHeight;
	};


	Position<float> getVecLineEndPoint(const VecLine& vl);

	float getLineLength(const Position<float>& p1, const Position<float>& p2);

	float getLineLength(const MyLine<float>& l);

	// obtient les informations trigo d'un triangle rectangle formé avec l'angle données
	TrigoInfo<float> getTriangleInfoFromHypo(const MyLine<float>& l, float angle, float h);

	void Translation(Position<float> *points, int size, float valueX, float valueY);

	void Reflect(Position<float> *points, int size, bool abs, bool ord);

	float generateFloat();


	float generateFloatInRange(float min, float max);
	uint generateUintInRange(uint min, uint max);

	inline std::mt19937 gen(std::random_device{}());
	void renderString(float x, float y, void* font, const char* str);

	void generateRandomColor(RGBColor<float>& randomColor);

	// genBuffer generer un buffer
	void genBuffer(GLuint* id, int position, int size, const void * data);

	void genIBOBuffer(uint* id, int position, int size, const void * data);

	Position<float> ConvertToNDC(int x, int y);

	void takeScreenShot();


	APPINFO BasicAppInfo();




	class GlutEngine {
		bool    isMax;
		int     windowId;
		int     wHeight, wWidth, sHeight, sWidth;
		bool    mode3d = false;


		void(*render)(void) = NULL;
		void(*mainloop)(int) = NULL;
		void(*keybinding)(unsigned char, int, int) = NULL;
		void(*funckeybinding)(int, int, int) = NULL;
		void(*mousebinding)(int, int, int, int) = NULL;
		void(*mouseMove)(int, int) = NULL;


	public:
		GlutEngine(int) {
			isMax = false;
		}
		GlutEngine(bool v) {
			mode3d = v;

		}
		bool Init(APPINFO, int, char**);
		void Run();

		void PutWindow(screenpositions position);
		void PutWindow(int x, int y);

		void ResizeBy(int x, int y);
		void SetFullScreen(bool on);

		void SetRenderFunc(void(*r)(void)) { render = r; }
		void SetMainFunc(void(*m)(int)) { mainloop = m; }
		void SetKeyFunc(void(*k)(unsigned char key, int x, int y)) { keybinding = k; }
		void SetFuncKeyFunc(void(*f)(int key, int x, int y)) { funckeybinding = f; }
		void SetMouseFunc(void(*m)(int, int, int, int)) { mousebinding = m; }
		void SetMouseMouveFunc(void(*m)(int, int)) { mouseMove = m; }

		void EndApp();

	private:
		bool InitGL();

	};
};
#endif // __ENGINE_H__
