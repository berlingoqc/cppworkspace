#include "engine.h"
#ifdef WITH_STB_IMAGE
	#define STB_IMAGE_IMPLEMENTATION
	#include <stb_image.h>
	#define STB_IMAGE_WRITE_IMPLEMENTATION
	#include <stb_image_write.h>
#endif
namespace ENGINE {
	Position<float> getVecLineEndPoint(const VecLine& vl) {
		return { vl.p.x + vl.v.x, vl.p.y + vl.v.y };
	}


	float getLineLength(const Position<float>& p1, const Position<float>& p2) {
		return glm::sqrt(glm::pow(p2.x - p1.x, 2) + glm::pow(p2.y - p1.y, 2));
	}


	float getLineLength(const MyLine<float>& l) {
		return getLineLength(l.p1, l.p2);
	}

	// obtient les informations trigo d'un triangle rectangle formé avec l'angle données
	TrigoInfo<float> getTriangleInfoFromHypo(const MyLine<float>& l, float angle, float h) {
		TrigoInfo<float> t;
		t.Angle = 180 - angle;
		t.H = h;
		float angleRadian = glm::radians(t.Angle);
		t.A = h * glm::sin(angleRadian);
		t.O = h * glm::cos(angleRadian);
		return t;
	}


	void Translation(Position<float> *points, int size, float valueX, float valueY) {
		for (int i = 0; i < size; i++) {
			points[i].x += valueX;
			points[i].y += valueY;
		}
	}

	void Reflect(Position<float> *points, int size, bool abs, bool ord) {
		for (int i = 0; i < size; i++) {
			if (abs)
				points[i].y = -points[i].y;
			if (ord)
				points[i].x = -points[i].x;
		}
	}


	float generateFloat() {
		return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}


	float generateFloatInRange(float min, float max) {
		assert(max > min);
		std::uniform_real_distribution<float> dis(min, max);
		return dis(gen);
	}
	uint generateUintInRange(uint min, uint max)
	{
		assert(max > min);
		std::uniform_real_distribution<float> dis(min, max);
		return (uint)dis(gen);
	}


	void renderString(float x, float y, void* font, const char* str) {
		glColor3f(0.0, 0.0, 0.0);
		glRasterPos2f(x, y);
		//glutBitmapString(font, (unsigned char*)str);
	}

	void generateRandomColor(RGBColor<float>& randomColor) {
		randomColor.R = generateFloat(); randomColor.G = generateFloat(); randomColor.B = generateFloat();
	}

	// genBuffer generer un buffer
	void genBuffer(GLuint* id, int position, int size, const void * data) {
		glGenBuffers(1, id); // Generer le VBO
		glBindBuffer(GL_ARRAY_BUFFER, *id);  // Lier le VBO
		glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW); // Définir la taille, les données et le type du VBO
	}

	void genIBOBuffer(uint* id, int position, int size, const void * data) {
		glGenBuffers(position, id);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *id);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
	}

	Position<float> ConvertToNDC(int x, int y) {
		Position<float> p;
		//int sizeY = glutGet(GLUT_WINDOW_HEIGHT);
		//int sizeX = glutGet(GLUT_WINDOW_WIDTH);
		//p.x = (x * (Maxndc - Minndc)) / sizeX + Minndc;
		//p.y = -(y - sizeY) * (Maxndc - Minndc) / sizeY + Minndc;
		return p;
	}

	void takeScreenShot() {
		int gW;//= glutGet(GLUT_WINDOW_WIDTH);
		int gH; //= glutGet(GLUT_WINDOW_HEIGHT);

		unsigned char* buffer = (unsigned char*)malloc(gW * gH * 3);
		glReadPixels(0, 0, gW, gH, GL_RGB, GL_UNSIGNED_BYTE, buffer);
		char name[512];
		long int t = time(NULL);
		sprintf(name, "screenshot_%ld.png", t);

		#ifdef WITH_STB_IMAGE
		unsigned char* last_row = buffer + (gW * 3 * (gH - 1));
		if (!stbi_write_png(name, gW, gH, 3, last_row, -3 * gW)) {
			std::cerr << "Error: could not write screenshot file " << name << std::endl;
		}
		#else
		std::cerr << "STB_IMAGE n'est pas définit impossible de capturer une image" << std::endl;
		#endif
	}

}



namespace ENGINE {


	bool GlutEngine::InitGL() {
		// Initialize la matrix de projection
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();

		// Initialize le mode Ortho pour pour changer le system de cordoner
		//glOrtho(0.0,wWidth,wHeight,0.0,1.0,-1.0);

		glPolygonMode(GL_FRONT, GL_FILL);

		// Initialize la matrix de modelview
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		// Initialize la couleur pour clean l'ecran
		glClearColor(0.f, 0.f, 0.f, 1.f);

		if (mode3d) {
			glEnable(GL_DEPTH_TEST);
		}

		// Valide s'il y a des erreurs
		GLenum error = glGetError();
		if (error != GL_NO_ERROR)
		{
			printf("Error initializing OpenGl! %s\n", gluErrorString(error));
			return false;
		}
		return true;
	}

	void GlutEngine::EndApp() {
		//glutDestroyWindow(windowId);
	}


	void GlutEngine::SetFullScreen(bool on) {
		if (on && !isMax) {
			//glutFullScreen();
			isMax = true;
		}
		else if (!on && isMax) {
			//glutReshapeWindow(wWidth, wHeight);
			isMax = false;
		}
	}

	void GlutEngine::ResizeBy(int x, int y) {
		if (isMax) return;
		wHeight += y;
		wWidth += x;

		//glutReshapeWindow(wWidth, wHeight);
	}

	void GlutEngine::PutWindow(screenpositions position) {
		if (isMax) return;
		int h, w;
		h = 0;
		w = 0;
		switch (position) {
		case center:
			//h = (sHeight / 2) - glutGet(GLUT_WINDOW_HEIGHT) / 2;
			//w = (sWidth / 2) - glutGet(GLUT_WINDOW_WIDTH) / 2;
			break;
		case bottomright:
			//w = glutGet(GLUT_SCREEN_WIDTH) - glutGet(GLUT_WINDOW_WIDTH);
			//h = glutGet(GLUT_SCREEN_HEIGHT) - glutGet(GLUT_WINDOW_HEIGHT);
			break;
		case topleft:
			h = 0;
			w = 0;
			break;
		case bottomleft:
			break;
		case topright:
			break;
		}
		//glutPositionWindow(w, h);
	}
	void GlutEngine::PutWindow(int x, int y)
	{
	}
	// InitGlutApp initialize une nouveau application avec Glut qui render simplement la scene de la function callback
	bool GlutEngine::Init(APPINFO info, int argc, char** argv) {
		//glutInit(&argc, argv); // Initialize GLUT chez pas ce que les arguments de la cmd font

		srand(static_cast <unsigned> (time(0)));
		// Crée une windows double buffer le gros
		if (mode3d) {
			//glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);

		}
		else {
			//glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);
		}

		wHeight = info.windowHeight;
		wWidth = info.windowWidth;

		//sHeight = glutGet(GLUT_SCREEN_HEIGHT);
		//sWidth = glutGet(GLUT_SCREEN_WIDTH);

		//glutInitWindowSize(info.windowWidth, info.windowHeight);   // Set la H&W de départ
		//glutInitWindowPosition(50, 50); // Set la position de départ
		//windowId = glutCreateWindow(info.title); // Crée notre windows avec un titre bien sur

		// Effectue l'init des shits d'opengl
		if (!InitGL()) {
			return false;
		}

		//glutKeyboardFunc(keybinding);
		//glutSpecialFunc(funckeybinding);
		//glutMouseFunc(mousebinding);
		//glutMotionFunc(mouseMove);
		//glutDisplayFunc(render); // Enregistre le callback pour le redraw

		glewInit();

		return true;
	}

	void GlutEngine::Run() {
		//glutTimerFunc(1000 / SCREEN_FPS, mainloop, 0);
		//glutMainLoop();
	}

	// BasicAppInfo retourne une structure rempli des info par default
	APPINFO BasicAppInfo() {
		APPINFO info;
		strcpy(info.title, "Demo");
		info.windowHeight = DEFAULT_SCREEN_HEIGHT;
		info.windowWidth = DEFAULT_SCREEN_WIDTH;
		return info;
	}
};