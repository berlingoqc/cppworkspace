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
	house.createBase(0.5f, 0.5f, 0.5f, glm::vec3(1.0f, 0.0f, 0.0f));
	glBindVertexArray(0);

	glBindVertexArray(vaoToitID);
	house.createCeiling(0.75f, 0.5f, 0.75f, glm::vec3(0.0f, 0.0f, 1.0f));
	glBindVertexArray(0);

	glBindVertexArray(vaoSolID);
	house.createFloor(10.0f, 20.0f, glm::vec3(0.4, 0.7, 0.3));
	glBindVertexArray(0);

}




struct ballinfo {
	glm::vec3 positions;
	float echelleZ;
	float echelleY;
	bool updown;
};

typedef std::vector<ballinfo>  listballinfo;

unsigned int vertexbuffer;
unsigned int ibo;

/* SECTION pour cr�e un carr� avec un IBO */
void createFaceIBO() {

	glm::vec3 vert[4]{ glm::vec3(-0.7f,0.7f,0.0f), glm::vec3(0.7f,0.7f,0.0f), glm::vec3(0.7f,-0.7f,0.0f), glm::vec3(-0.7f,-0.7f,0.0f) };

	// Cr�e notre vertex buffer
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer); // GL_ARRAY_BUFFER Attributs de sommets (dont la position)
	glBufferData(GL_ARRAY_BUFFER, sizeof(vert), vert, GL_STATIC_DRAW);

	// Cr�e notre Index Buffer object pour dessiner les deux triangles avec nos 4 points
	unsigned int i[]{ 0,1,2,0,2,3 };
	glGenBuffers(1, &ibo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo); // ELEMENT_ARRAY_BUFFER repr�sente Indices d'un table de sommet
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(i), i, GL_STATIC_DRAW);

}

/* Cr�e et dessine le sol */
void createDrawFloor() {
	unsigned int vbo;

	glm::vec3 sol[8]{
		glm::vec3(1.0f,-1.0f,-100.0f), glm::vec3(0.0,1.0,1.0),
		glm::vec3(1.0,-1.0,1.0), glm::vec3(0.0,1.0,0.0),
		glm::vec3(-1.0,-1.0,-100.0), glm::vec3(0.0,1.0,0.0),
		glm::vec3(-1.0,-1.0,1.0), glm::vec3(0.0,1.0,0.0)
	};

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(sol), sol, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), 0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}


void displayRect() {
	// clear le background avec la couleur voulu
	glClearColor(255, 255, 255, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// bind notre programme de shader
	glUseProgram(ShaderID);


	glEnableVertexAttribArray(0);;
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	glDisableVertexAttribArray(0);

	glutSwapBuffers();

}




void display() {
	// Cr�e nos matrice de transformation
	glm::mat4x4 vue = glm::mat4(1.0);
	glm::mat4x4 proj = glm::mat4(1.0);
	glm::mat4x4 trans = glm::mat4(1.0);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Matrice de vue qu'on utilise pour gerer l'angle de vue sur la scene
	vue = glm::lookAt(glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, 0.0, -20.0), glm::vec3(0.0, 1.0, 0.0));
	glUniformMatrix4fv(VueID, 1, GL_FALSE, &vue[0][0]);

	// matrice de projection ( a definir )
	proj = glm::perspective(glm::radians(45.0f), static_cast<float>(DEFAULT_SCREEN_WIDTH / DEFAULT_SCREEN_HEIGHT), 0.1f, 100.0f);
	glUniformMatrix4fv(ProjectionID, 1, GL_FALSE, &proj[0][0]);

	displayRect();

	// Cr�e le sol une premiere fois pour le bas
	//glUniformMatrix4fv(TranslationID, 1, GL_FALSE, &trans[0][0]);
	//createDrawFloor();

	// Change la matrice de translation pour le redessiner dans le haut plafond
	//trans = glm::translate(trans, glm::vec3(0.0, 2.0, 0.0));
	//glUniformMatrix4fv(TranslationID, 1, GL_FALSE, &trans[0][0]);
	//createDrawFloor();




}


void traitementMenuSettings(int value) {
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



	// Init du menu contextuel
	createMenu();

	if (!initShaderHouse()) {
		std::cerr << "Echer de l'initialisation des shaders" << std::endl;
		return -1;
	}

	initHouse();


	app->Run();

	return 0;
}


