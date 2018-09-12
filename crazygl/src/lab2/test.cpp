#include "../../include/engine.hpp"
#include "../../include/shaders.hpp"
using namespace ENGINE;

GlutEngine* app;


unsigned int ShaderID;


// Display mon quad dans une couleur qui change selon le keyboard
void display() {
    glClearColor(0.2f,0.3f,0.3f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    glUseProgram(ShaderID);

    // update la couleur uniforme
    float value = generateFloat(); 
    float greenValue
    
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
       return -1;
   }
   ShaderID = MyShader.GetShaderID();

   app->Run();
   return 0;
}
