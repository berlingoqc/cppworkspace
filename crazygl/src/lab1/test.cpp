#include "../../include/engine.h"



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
            engine::EndGlutApp();
        break;
    }
}

void specialkeybinding(int key, int x , int y) {
    switch(key) {

        // maximise la fenetre
        case GLUT_KEY_UP:
            engine::ToggleFullscreen(true);
        break;
        // revenir a la grandeur original
        case GLUT_KEY_DOWN:
            engine::ToggleFullscreen(false);
        break;
        // agrandir la fenetre de 50px
        case GLUT_KEY_RIGHT:
            engine::ResizeBy(50);
        break;
        // diminiuer la fenetre de 50px
        case GLUT_KEY_LEFT:
            engine::ResizeBy(-50);
        break;
        // deplacer la fenetre dans le coin superieur gauche
        case GLUT_KEY_F1:
            glutPositionWindow(0,0);
        break;
        // deplacer la fenetre dans le coin inferieur droit
        case GLUT_KEY_F2:
            engine::PutBottomRightCorner(); 
        break;
        // deplace la fenetre parfaitement au centre
        case GLUT_KEY_F3:
            engine::CenterScreen();
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
   engine::APPINFO info = engine::BasicAppInfo();
   engine::InitGlutApp( info, argc, argv, display, mainLoop,keybinding,specialkeybinding);
   return 0;
}
