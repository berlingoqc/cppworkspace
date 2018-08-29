#include "../../include/engine.h"


// La couleur courante pour render mon quad
int gColorMode = COLOR_MODE_CYAN;
GLfloat gProjectionScale = 1.f;


// Display mon quad dans une couleur qui change selon le keyboard
void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    // Reset la matrix modelview
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Dépalace au centre de l'écran
    glTranslatef(SCREEN_WIDTH/2.f,SCREEN_HEIGHT/2.f,0.f);

    // Render le quad
    if(gColorMode == COLOR_MODE_CYAN) {
        glBegin(GL_QUADS);
            glColor3f(0.f,1.f,1.f);
            glVertex2f(-50.f,-50.f);
            glVertex2f(50.f,-50.f);
            glVertex2f(50.f,50.f);
            glVertex2f(-50.f,50.f);
        glEnd();
    } else if (gColorMode == COLOR_MODE_MULTI) {
        glBegin(GL_QUADS);
            glColor3f(1.f,0.f,0.f); glVertex2f(-50.f,-50.f);
            glColor3f(1.f,1.f,0.f); glVertex2f(50.f,-50.f);
            glColor3f(1.f,1.f,1.f); glVertex2f(50.f,50.f);
            glColor3f(0.f,0.f,1.f); glVertex2f(-50.f,50.f);
        glEnd();
    }
    // update screen
    glutSwapBuffers();
}

void keybinding(unsigned char key,int x,int y) {
    if(key == 'q') {
        // Change de mode de couleur
        if(gColorMode == COLOR_MODE_CYAN)
            gColorMode = COLOR_MODE_MULTI;
        else
            gColorMode = COLOR_MODE_CYAN;
    }
    else if ( key == 'e'){
        if(gProjectionScale == 1.f){
            gProjectionScale = 2.f;
        } else if (gProjectionScale == 2.f) {
            gProjectionScale = 0.5f;
        } else if(gProjectionScale == 0.5f){
            gProjectionScale = 1.f;
        }

        // update la matrix de projection
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0,SCREEN_WIDTH*gProjectionScale,SCREEN_WIDTH*gProjectionScale,0.0,1.0,-1.0);
    }
}


void mainLoop(int val) {
    display();
    // Roule la frame une autre fois
    glutTimerFunc(1000/SCREEN_FPS,mainLoop,val);
}

int main(int argc,char** argv) {
   engine::APPINFO info = engine::BasicAppInfo();
   engine::InitGlutApp( info, argc, argv, display, mainLoop,keybinding);
   return 0;
}
