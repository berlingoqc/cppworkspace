#include "engine.h"
#include "shaders.h"
#include "city.h"
#include "camera.h"

using namespace ENGINE;

// DEFINITON VARIABLE GLOBALE
typedef unsigned int uint;

enum BASIC_OPTION_MENU { SAVE_DRAWING, EXIT_APP };


ProceduralCity	city;


void traitementMenuPrincipal(int value) {

	if (value == SAVE_DRAWING)
		takeScreenShot();
}

void createMenu() {
	int menuPrincipal;


	menuPrincipal = glutCreateMenu(traitementMenuPrincipal);
	glutAddMenuEntry("Sauvegarder", SAVE_DRAWING);
	glutAddMenuEntry("Exit", EXIT_APP);

	glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void mouse(int x, int y)
{
	city.getCamera().mouse_move(x, y);
}

void keyboard(uchar btn, int x, int y)
{
	if (btn == 't')
		takeScreenShot();
	city.getCamera().keyboard_press(btn, x, y);
}

void keyboard_release(uchar btn, int x, int y)
{
	city.getCamera().keyboard_release(btn, x, y);
}

void mouse_roll(int roue,int dir, int x,int y)
{
	if(dir == 1)
	{
		city.getCamera().addFOV(-1.0f);
	} else
	{
		city.getCamera().addFOV(1.0f);
	}

	glutPostRedisplay();
	
}

void render()
{

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
;
	
	city.render();

	glutSwapBuffers();
	glutPostRedisplay();
}

void close()
{
	glutLeaveMainLoop();
}

void mainLoop(int val) {
	glutPostRedisplay();
	glutTimerFunc(1000 / SCREEN_FPS, mainLoop, val);
}

int main(int argc, char** argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(glutGet(GLUT_SCREEN_WIDTH), glutGet(GLUT_SCREEN_HEIGHT));
	glutCreateWindow("Procedural City");
	glutFullScreen();
	/*******************/
	glewInit();

	float z = 1.0f;
	glutDisplayFunc(render);
	glutCloseFunc(close);
	glutKeyboardFunc(keyboard);
	glutPassiveMotionFunc(mouse);
	glutKeyboardUpFunc(keyboard_release);
	glutIdleFunc(render);
	glutMouseWheelFunc(mouse_roll);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
	glutSetCursor(GLUT_CURSOR_NONE);

	createMenu();


	if(!city.configure("textures"))
	{
		return 1;
	}

	glutMainLoop();


	return 0;
}

