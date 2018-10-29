#include "../../include/engine.hpp"
#include "../../include/shaders.hpp"


using namespace ENGINE;

// DEFINITON VARIABLE GLOBALE


enum BASIC_OPTION_MENU { SAVE_DRAWING, EXIT_APP };

// Variable pour l'application glut
GlutEngine* app;

// Variable de l'identifiant de mon program de shader
unsigned int ShaderID;


// Display mon quad dans une couleur qui change selon le keyboard
void display() {
    // clear le background avec la couleur voulu
    glClearColor(255,255,255,1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    // bind notre programme de shader
   	glUseProgram(ShaderID);

    glLineWidth(3.0f);


	glFlush();
}


void traitementMenuSettings(int value) {
}

void traitementMenuPrincipal(int value) {
}

void createMenu() {
    int menuPrincipal, menuOption;

    menuOption = glutCreateMenu(traitementMenuSettings);

    menuPrincipal = glutCreateMenu(traitementMenuPrincipal);
    glutAddSubMenu("Options",menuOption);
    glutAddMenuEntry("Sauvegarder", SAVE_DRAWING);
    glutAddMenuEntry("Exit",EXIT_APP);

    glutAttachMenu(GLUT_RIGHT_BUTTON);
}


void keybinding(unsigned char key,int x,int y) {
}

void specialkeybinding(int key,int x,int y) {

}

void mousebinding(int x,int y,int z, int a) {

}

void mouseMove(int x,int y) {

}

void mainLoop(int val) {
    glutPostRedisplay();
    glutTimerFunc(1000/SCREEN_FPS,mainLoop,val);
}



int main(int argc,char** argv) {

   ENGINE::APPINFO info = ENGINE::BasicAppInfo();
   GlutEngine g(0);
   app = &g;
   app->SetMainFunc(mainLoop);
   app->SetRenderFunc(display);
   app->SetKeyFunc(keybinding);
   app->SetFuncKeyFunc(specialkeybinding);
   app->SetMouseFunc(mousebinding);
   app->SetMouseMouveFunc(mouseMove);
   app->Init(info,argc,argv);


   // Initialize no MyShaders
   ENGINE::MyShader MyShader;
   if(!MyShader.OpenMyShader("vertex.glsl","fragment.glsl")) {
       return -1;
   }
   ShaderID = MyShader.GetShaderID();

   // Init du menu contextuel
   createMenu();

   app->Run();

   return 0;
}
