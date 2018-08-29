#ifndef __ENGINE_H__
#define __ENGINE_H__

// Import des header de opengl
#include <GL/glew.h>
#include <GL/glut.h>

// Import des autres headers 
#include <stdio.h>
#include <string.h>
#include <math.h>

// Constante de l'écran
const int SCREEN_WIDTH  = 640;
const int SCREEN_HEIGHT = 480;
const int SCREEN_FPS    = 60;


// Mode de couleur
const int COLOR_MODE_CYAN = 0;
const int COLOR_MODE_MULTI = 1;

bool isMax = false;

int windowId;

int ScreenHeight = SCREEN_HEIGHT;
int ScreenWidth = SCREEN_WIDTH;

int MiddleWidth;
int MiddleHeight;

namespace engine
{
    // APPINFO est une structure qui contient les informations relatives au démarre d'une nouvelle application OpenGL
    struct APPINFO {
                char title[128];
                int windowWidth;
                int windowHeight;

                union
                {
                    struct
                    {
                        unsigned int    fullscreen :1;
                        unsigned int    vsync      :1;
                        unsigned int    debug      :1;
                    };
                    unsigned int        all;
                } flags;

    };


    bool InitGL() {
        // Initialize la matrix de projection
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        // Initialize le mode Ortho pour pour changer le system de cordoner
        glOrtho(0.0,SCREEN_WIDTH,SCREEN_HEIGHT,0.0,1.0,-1.0);

        // Initialize la matrix de modelview
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // Initialize la couleur pour clean l'ecran
        glClearColor(0.f,0.f,0.f,1.f);

        // Valide s'il y a des erreurs
        GLenum error = glGetError();
        if ( error != GL_NO_ERROR )
        {
            printf("Error initializing OpenGl! %s\n", gluErrorString(error));
            return false;
        }
        return true;
    }

    void EndGlutApp() {
        glutDestroyWindow(windowId);
    }


    void ToggleFullscreen(bool on) {
        if(on && !isMax) {
            glutFullScreen();
            isMax = true;
        } else if(!on && isMax) {
            glutReshapeWindow(ScreenWidth,ScreenHeight);
            isMax = false;
        }
    }

    void ResizeBy(int px) {
        ScreenHeight    += px;
        ScreenWidth     += px;

        glutReshapeWindow(ScreenWidth,ScreenHeight);
    }

    void CenterScreen() {
        int h = MiddleHeight - glutGet(GLUT_WINDOW_HEIGHT) /2;
        int w = glutGet(GLUT_WINDOW_WIDTH) / 2;
        w = MiddleWidth - w;

        glutPositionWindow(w,h);
    }

    void PutBottomRightCorner() {
        int width = glutGet(GLUT_SCREEN_WIDTH);
        int height = glutGet(GLUT_SCREEN_HEIGHT);
        int wWidth = glutGet(GLUT_WINDOW_WIDTH);
        int wHeight = glutGet(GLUT_WINDOW_HEIGHT);

        glutPositionWindow(width-wWidth,height-wHeight);
    }

    // InitGlutApp initialize une nouveau application avec Glut qui render simplement la scene de la function callback
    void InitGlutApp(APPINFO info,int argc,char** argv, void (*display)( void ), void (*mainloop)(int), void (*keybinding)(unsigned char,int,int),void (*specialKeyBinging)(int,int,int)) {
        glutInit(&argc, argv); // Initialize GLUT chez pas ce que les arguments de la cmd font

        // Crée une windows double buffer le gros
        glutInitDisplayMode(GLUT_DOUBLE);

        ScreenHeight = info.windowHeight;
        ScreenWidth = info.windowWidth;
        glutInitWindowSize(info.windowWidth, info.windowHeight);   // Set la H&W de départ
        glutInitWindowPosition(50, 50); // Set la position de départ
        windowId = glutCreateWindow(info.title); // Crée notre windows avec un titre bien sur

        // Effectue l'init des shits d'opengl
        if (!InitGL()) {
            return;
        }

        glutKeyboardFunc(keybinding);
        glutSpecialFunc(specialKeyBinging);

        glutDisplayFunc(display); // Enregistre le callback pour le redraw

        glutTimerFunc(1000/SCREEN_FPS, mainloop,0);

        glutMainLoop();           // Enter the infinitely event-processing loop
    }

    // BasicAppInfo retourne une structure rempli des info par default
    APPINFO BasicAppInfo() {
        APPINFO info;
        strcpy(info.title,"Demo");
        info.windowHeight = SCREEN_HEIGHT;
        info.windowWidth = SCREEN_WIDTH;
        return info;
    }
};

#endif // __ENGINE_H__
