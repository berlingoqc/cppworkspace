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
	city.render();
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
	glClearColor(0.0f, 0.4f, 0.5f, 0.0f);
	glutSetCursor(GLUT_CURSOR_NONE);

	createMenu();

	if(!city.configure("textures"))
	{
		return 1;
	}

	glutMainLoop();


	return 0;
}

