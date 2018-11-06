#include "engine.h"
#include "shaders.h"
#include "house_maker.h"

using namespace ENGINE;

// DEFINITON VARIABLE GLOBALE
typedef unsigned int uint;

enum BASIC_OPTION_MENU { SAVE_DRAWING, EXIT_APP };


// Variable pour l'application glut
GlutEngine* app;

// Variable global pour mes handlers OpenGL
unsigned int ShaderID;
unsigned int VueID;
unsigned int ProjectionID;
unsigned int TranslationID;

//  Handler globaux pour ma maison
uint vaoBaseID;
uint vaoToitID;
uint vaoSolID;
uint rotationID;
uint echelleID;

// Variables globales pour la maison
float	houseScale = 1.0f;
float	houseRotX = 0.0f;
float   houseRotY = 0.0f;
int		firstPass = 0;



void renderHouseScene() {

	glm::mat4x4 rot = glm::mat4(1.0);
	glm::mat4x4 ech = glm::mat4(1.0);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	rot = glm::rotate(rot, glm::radians(houseRotX), glm::vec3(1.0, 0.0, 0.0));
	rot = glm::rotate(rot, glm::radians(houseRotY), glm::vec3(0.0, 1.0, 0.0));

	ech = glm::scale(ech, glm::vec3(houseScale, houseScale, houseScale));

	glUniformMatrix4fv(rotationID, 1, GL_FALSE, &rot[0][0]);
	glUniformMatrix4fv(echelleID, 1, GL_FALSE, &ech[0][0]);

	glBindVertexArray(vaoBaseID);
	glDrawElements(GL_TRIANGLES, 14 * 3, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);

	glBindVertexArray(vaoToitID);
	glDrawElements(GL_TRIANGLES, 6 * 3, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);

	glBindVertexArray(vaoSolID);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);


	glutSwapBuffers();
}

bool initShaderHouse() {
	// Initialize no MyShaders
	ENGINE::MyShader MyShader;
	if (!MyShader.OpenMyShader("vertex.glsl", "fragment.glsl")) {
		std::cerr << "Erreur dans l'ouverture des shaders" << std::endl;
		return false;
	}
	ShaderID = MyShader.GetShaderID();
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glUseProgram(ShaderID);
	// Get les handlers des variables globales 
	rotationID = glGetUniformLocation(ShaderID, "gRot");
	assert(rotationID != 0xFFFFFFFF);

	echelleID = glGetUniformLocation(ShaderID, "gScale");
	assert(echelleID != 0xFFFFFFFF);

	glGenVertexArrays(1, &vaoBaseID);
	glGenVertexArrays(1, &vaoToitID);
	glGenVertexArrays(1, &vaoSolID);


	return true;
}


void initHouse() {
	House::House_Maker house;

	glBindVertexArray(vaoBaseID);
	house.creeBase(0.5f, 0.5f, 0.5f, glm::vec3(1.0f, 0.0f, 0.0f));
	glBindVertexArray(0);

	glBindVertexArray(vaoToitID);
	house.creeToit(0.75f, 0.5f, 0.75f, glm::vec3(0.0f, 0.0f, 1.0f));
	glBindVertexArray(0);

	glBindVertexArray(vaoSolID);
	house.creeSol(10.0f, 20.0f, glm::vec3(0.4, 0.7, 0.3));
	glBindVertexArray(0);

}

void traitementMenuPrincipal(int value) {
}

void createMenu() {
	int menuPrincipal;


	menuPrincipal = glutCreateMenu(traitementMenuPrincipal);
	glutAddMenuEntry("Sauvegarder", SAVE_DRAWING);
	glutAddMenuEntry("Exit", EXIT_APP);

	glutAttachMenu(GLUT_RIGHT_BUTTON);
}


void keybinding(unsigned char key, int x, int y) {
}

void specialkeybinding(int key, int x, int y) {

}

void mousebinding(int x, int y, int z, int a) {

}

void mouseMove(int x, int y) {

}

void mainLoop(int val) {
	glutPostRedisplay();
	glutTimerFunc(1000 / SCREEN_FPS, mainLoop, val);
}

int main(int argc, char** argv) {

	ENGINE::APPINFO info = ENGINE::BasicAppInfo();
	GlutEngine g(true);
	app = &g;
	app->SetMainFunc(mainLoop);
	app->SetRenderFunc(renderHouseScene);
	app->SetKeyFunc(keybinding);
	app->SetFuncKeyFunc(specialkeybinding);
	app->SetMouseFunc(mousebinding);
	app->SetMouseMouveFunc(mouseMove);
	app->Init(info, argc, argv);

	if (!initShaderHouse()) {
		std::cerr << "Echer de l'initialisation des shaders" << std::endl;
		return -1;
	}
	initHouse();

	// Init du menu contextuel
	createMenu();


	app->Run();

	return 0;
}

