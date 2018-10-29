
#include "engine.h"

#include "shaders.h"

using namespace ENGINE;

// Variable globale pour mon engine (wrapper autour de glut)
GlutEngine* app;

// Variable pour l'identifiant de mon program de shader compiler
unsigned int ShaderID;

// Vector content la liste de point utilisé pour render les formes a l'écran
std::vector<Position<float>> listPoint;

enum OptionMenu { CLEAR_SCREEN, DRAW_POINTS, DRAW_LINES, DRAW_TRIANGLE, DRAW_QUADS, DRAW_CON_LINES, EXIT_APP };
enum OptionColor { RED, GREEN, BLUE, YELLOW, RANDOM };


void createDrawing() {
	GLfloat vertices[86]{
		-1.0, 1.0, // Triangle top left
		-1.0, 0.5,
		-0.5, 1.0,

		 0.0, 0.0, // Triangle milieu haut droit
		 0.5, 0.0,
		 0.0, 0.5,

		 0.0, 0.0, // Triangle milieu bas droit
		 0.0f, -0.5f,
		 0.5f, 0.0f,

		 0.0, 0.0, // Triangle milieu bas gauche
		 -0.5, 0.0f,
		 0.0f, -0.5f,

		 0.0f, 0.0f, // Triangle milieu haut gauche
		 0.0f, 0.5f,
		 -0.5f,0.0f,

		 0.5f, 1.0f, // Triangle top right
		 1.0f, 0.5f,
		 1.0f, 1.0f,

		 -1.0f,-1.0f, // Triangle bot left
		 -0.5f,-1.0f,
		 -1.0f, -0.5f,

		 0.5f, -1.0f, // Triangle bot right
		 1.0f, -1.0f,
		 1.0f, -0.5f,

		 -0.25f,1.0f, // Triangle milieu top
		 0.0f, 0.75f,
		 0.25f, 1.0f,

		-0.25f,-1.0f, // Triangle milieu bot
		0.25f, -1.0f,
		0.0f, -0.75f,

		0.0f, -0.5f,
		0.0f, -0.75f,
		0.25f, -0.625f,

		0.0f, -0.5f,
		0.0f, -0.75f,
		-0.25f, -0.625f,

		0.0f, 0.5f,
		0.0f, 0.75f,
		0.25f, 0.625f,

		0.0f, 0.5f,
		0.0f, 0.75f,
		-0.25f, 0.625f,
	};
	GLuint vbo;
	genBuffer(&vbo, 0, sizeof(vertices), vertices);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
}

void createDrawingLine() {
	GLuint vboID;

	GLfloat sommets[8] = {
		0.0f,0.0f,0.5f,0.5f,
		0.0f,-0.5f,0.0f,0.5f,
	};
	genBuffer(&vboID, 0, sizeof(sommets), sommets);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0); // Définit le pointeur d'attributs des sommets
}

void createPointsClick() {
	GLuint vboID;

	genBuffer(&vboID, 0, listPoint.size() * sizeof(Position<float>), listPoint.data());
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
}


// Display mon quad dans une couleur qui change selon le keyboard
void display() {
	glClearColor(255.0f, 255.0f, 255.f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glUseProgram(ShaderID);

	glPointSize(10.0);

	createDrawing();
	glDrawArrays(GL_TRIANGLES, 0, 42);


	glDisableVertexAttribArray(0);

	glFlush();
}

void mousebinding(int button, int state, int x, int y) {
	Position<float> p = ENGINE::ConvertToNDC(x, y);
	if (button == 0 && state == 0) {
		listPoint.push_back(p);
	}
}

void keybinding(unsigned char key, int x, int y) {

}
void specialkeybinding(int key, int x, int y) {

}


void mainLoop(int val) {
	glutPostRedisplay();
	// Roule la frame une autre fois
	glutTimerFunc(1000 / SCREEN_FPS, mainLoop, val);
}

int main(int argc, char** argv) {
	ENGINE::APPINFO info = ENGINE::BasicAppInfo();
	GlutEngine g(0);
	app = &g;
	app->SetMainFunc(mainLoop);
	app->SetRenderFunc(display);
	app->SetKeyFunc(keybinding);
	app->SetFuncKeyFunc(specialkeybinding);
	app->SetMouseFunc(mousebinding);
	app->Init(info, argc, argv);


	// Initialize no MyShaders
	ENGINE::MyShader MyShader;
	if (!MyShader.OpenMyShader("vertex.glsl", "fragment.glsl")) {
		return -1;
	}
	ShaderID = MyShader.GetShaderID();


	app->Run();
	return 0;
}
