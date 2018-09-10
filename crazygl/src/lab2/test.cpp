#include "../../include/engine.hpp"
#include "../../include/shaders.hpp"
using namespace ENGINE;

GlutEngine* app;



// Display mon quad dans une couleur qui change selon le keyboard
void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    // update screen
    glutSwapBuffers();
}

void keybinding(unsigned char key,int x,int y) {

}
void specialkeybinding(int key, int x , int y) {

}

void mainLoop(int val) {
    glutPostRedisplay();
    // Roule la frame une autre fois
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
   app->Init(info,argc,argv);


    // Initialize no MyShaders
    ENGINE::MyShader MyShader;
    if(!MyShader.OpenMyShader("vertex.glsl","fragment.glsl")) {
        MyShader.PrintErrorStack();
        return -1;
    }



   app->Run();
   return 0;
}
