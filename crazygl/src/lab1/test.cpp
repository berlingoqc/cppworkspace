#include "../../include/engine.hpp"

using namespace ENGINE;

GlutEngine* app;



float generateFloat() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

// Display mon quad dans une couleur qui change selon le keyboard
void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    // update screen
    glutSwapBuffers();
}

void keybinding(unsigned char key,int x,int y) {
    switch(key) {
        case 'r' : glClearColor(1.0f,0.f,0.f,0.f); break;
        case 'b' : glClearColor(0.f,0.f,1.f,1.f); break;
        case 'v' : glClearColor(0.f,1.f,0.f,1.f); break;
        case 'n' : glClearColor(0.f,0.f,0.f,1.f); break;
        case 'm' : glClearColor(1.f,0.f,1.f,1.f); break;
        case 't' : glClearColor(0.5f,1.f,1.f,1.f); break;
        case 'o' : glClearColor(1.f,0.5f,0.f,1.f); break;
        case 'a' : glClearColor(1.f,1.f,1.f,0.f); break;
        // ASCII 27 est echape et on quitte l'application
        case 27 :
            app->EndApp();
        break;
    }
}

void specialkeybinding(int key, int x , int y) {
    switch(key) {

        // maximise la fenetre
        case GLUT_KEY_UP:
            app->SetFullScreen(true);
        break;
        // revenir a la grandeur original
        case GLUT_KEY_DOWN:
            app->SetFullScreen(false);
        break;
        // agrandir la fenetre de 50px
        case GLUT_KEY_RIGHT:
            app->ResizeBy(50,50);
        break;
        // diminiuer la fenetre de 50px
        case GLUT_KEY_LEFT:
            app->ResizeBy(-50,-50);
        break;
        // deplacer la fenetre dans le coin superieur gauche
        case GLUT_KEY_F1:
            app->PutWindow(topleft);
        break;
        // deplacer la fenetre dans le coin inferieur droit
        case GLUT_KEY_F2:
            app->PutWindow(bottomright);
        break;
        // deplace la fenetre parfaitement au centre
        case GLUT_KEY_F3:
            app->PutWindow(center);
        break;
        // genere une nouvelle couleur aleatoire comme fond d'ecran
        case GLUT_KEY_F4:
            float f1 = generateFloat();
            float f2 = generateFloat();
            float f3 = generateFloat();
            glClearColor(f1,f2,f3,1.0f);
        break;
    }
}

void mainLoop(int val) {
    glutPostRedisplay();
    // Roule la frame une autre fois
    glutTimerFunc(1000/SCREEN_FPS,mainLoop,val);
}

int main(int argc,char** argv) {
   ENGINE::APPINFO info = engine::BasicAppInfo();
   GlutEngine g(0);
   app = &g;
   app->SetMainFunc(mainLoop);
   app->SetRenderFunc(display);
   app->SetKeyFunc(keybinding);
   app->SetFuncKeyFunc(specialkeybinding);
   app->Init(info,argc,argv);
   app-­>Run();
   return 0;
}
