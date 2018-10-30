#include "engine.h"
#include "shaders.h"



using namespace ENGINE;

// DEFINITON VARIABLE GLOBALE


enum BASIC_OPTION_MENU { SAVE_DRAWING, EXIT_APP };

// Variable pour l'application glut
GlutEngine* app;

// Variable global pour mes handlers OpenGL
unsigned int ShaderID;
unsigned int VueID;
unsigned int ProjectionID;
unsigned int TranslationID;


struct ballinfo {
	glm::vec3 positions;
	float echelleZ;
	float echelleY;
	bool updown;
};

typedef std::vector<ballinfo>  listballinfo;



unsigned int vertexbuffer;
unsigned int ibo;

/* SECTION pour crée un carré avec un IBO */
void createFaceIBO() {

	glm::vec3 vert[4]{ glm::vec3(-0.7f,0.7f,0.0f), glm::vec3(0.7f,0.7f,0.0f), glm::vec3(0.7f,-0.7f,0.0f), glm::vec3(-0.7f,-0.7f,0.0f) };

	// Crée notre vertex buffer
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer); // GL_ARRAY_BUFFER Attributs de sommets (dont la position)
	glBufferData(GL_ARRAY_BUFFER, sizeof(vert), vert, GL_STATIC_DRAW);

	// Crée notre Index Buffer object pour dessiner les deux triangles avec nos 4 points
	unsigned int i[]{ 0,1,2,0,2,3 };
	glGenBuffers(1, &ibo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo); // ELEMENT_ARRAY_BUFFER représente Indices d'un table de sommet
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(i), i, GL_STATIC_DRAW);

}

/* Crée et dessine le sol */
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


void display() {
	// Crée nos matrice de transformation
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

	// Crée le sol une premiere fois pour le bas
	glUniformMatrix4fv(TranslationID, 1, GL_FALSE, &trans[0][0]);
	createDrawFloor();

	// Change la matrice de translation pour le redessiner dans le haut plafond
	trans = glm::translate(trans, glm::vec3(0.0, 2.0, 0.0));
	glUniformMatrix4fv(TranslationID, 1, GL_FALSE, &trans[0][0]);
	createDrawFloor();




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
	app->SetRenderFunc(display);
	app->SetKeyFunc(keybinding);
	app->SetFuncKeyFunc(specialkeybinding);
	app->SetMouseFunc(mousebinding);
	app->SetMouseMouveFunc(mouseMove);
	app->Init(info, argc, argv);


	// Initialize no MyShaders
	ENGINE::MyShader MyShader;
	if (!MyShader.OpenMyShader("vertex.glsl", "fragment.glsl")) {
		return -1;
	}
	ShaderID = MyShader.GetShaderID();

	// Init du menu contextuel
	createMenu();

	createFaceIBO();

	app->Run();

	return 0;
}


