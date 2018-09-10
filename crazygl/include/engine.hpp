#ifndef __ENGINE_H__
#define __ENGINE_H__

#include "headers.hpp"

// Constante de l'écran
const int DEFAULT_SCREEN_WIDTH  = 640;
const int DEFAULT_SCREEN_HEIGHT = 480;
const int SCREEN_FPS    = 60;


namespace ENGINE
{
    // enum screenpositions contient les positions possible pour placer rapidement notre fenetre
    enum screenpositions { topleft, topright, center, bottomleft, bottomright };

    // APPINFO est une structure qui contient les informations relatives au démarre d'une nouvelle application OpenGL
    struct APPINFO {
            char title[128];
            int windowWidth;
            int windowHeight;
    };

    class GlutEngine {
        bool    isMax;
        int     windowId;
        int     wHeight, wWidth, sHeight, sWidth;


        void(*render)(void);
        void(*mainloop)(int);
        void(*keybinding)(unsigned char,int,int);
        void(*funckeybinding)(int,int,int);


        public:
            GlutEngine(int) {
                isMax = false;
            }
            bool Init(APPINFO,int,char**);    
            void Run();

            void PutWindow(screenpositions position);
            void PutWindow(int x, int y);

            void ResizeBy(int x,int y);
            void SetFullScreen(bool on);
            
            void SetRenderFunc(void(*r)(void)) { render = r; }
            void SetMainFunc(void (*m)(int)) { mainloop = m;}
            void SetKeyFunc(void(*k)(unsigned char key,int x , int y)) { keybinding = k; }
            void SetFuncKeyFunc(void(*f)(int key, int x,int y)) { funckeybinding = f; }
      
            void EndApp();

        private:
            bool InitGL();
    
    };


    bool GlutEngine::InitGL() {
        // Initialize la matrix de projection
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        // Initialize le mode Ortho pour pour changer le system de cordoner
        glOrtho(0.0,wWidth,wHeight,0.0,1.0,-1.0);

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

    void GlutEngine::EndApp() {
        glutDestroyWindow(windowId);
    }


    void GlutEngine::SetFullScreen(bool on) {
        if(on && !isMax) {
            glutFullScreen();
            isMax = true;
        } else if(!on && isMax) {
            glutReshapeWindow(wWidth,wHeight);
            isMax = false;
        }
    }

    void GlutEngine::ResizeBy(int x,int y) {
        if(isMax) return;
        wHeight    += y;
        wWidth     += x;

        glutReshapeWindow(wWidth,wHeight);
    }

    void GlutEngine::PutWindow(screenpositions position) {
        if(isMax) return;
        int h, w;
        h = 0;
        w = 0;
        switch(position) {
            case center:
                h = (sHeight/2) - glutGet(GLUT_WINDOW_HEIGHT) /2;
                w = (sWidth/2) - glutGet(GLUT_WINDOW_WIDTH) / 2;
            break;
            case bottomright:
                w = glutGet(GLUT_SCREEN_WIDTH) - glutGet(GLUT_WINDOW_WIDTH);
                h = glutGet(GLUT_SCREEN_HEIGHT) - glutGet(GLUT_WINDOW_HEIGHT);
            break;
            case topleft:
                h = 0;
                w = 0;
            break;
            case bottomleft:
            break;
            case topright:
            break;
        }
        glutPositionWindow(w,h);
    }
    // InitGlutApp initialize une nouveau application avec Glut qui render simplement la scene de la function callback
    bool GlutEngine::Init(APPINFO info,int argc,char** argv) {
        glutInit(&argc, argv); // Initialize GLUT chez pas ce que les arguments de la cmd font

        // Crée une windows double buffer le gros
        glutInitDisplayMode(GLUT_DOUBLE);

        wHeight = info.windowHeight;
        wWidth = info.windowWidth;

        sHeight = glutGet(GLUT_SCREEN_HEIGHT);
        sWidth = glutGet(GLUT_SCREEN_WIDTH);

        glutInitWindowSize(info.windowWidth, info.windowHeight);   // Set la H&W de départ
        glutInitWindowPosition(50, 50); // Set la position de départ
        windowId = glutCreateWindow(info.title); // Crée notre windows avec un titre bien sur

        // Effectue l'init des shits d'opengl
        if (!InitGL()) {
            return false;
        }

        glutKeyboardFunc(keybinding);
        glutSpecialFunc(funckeybinding);

        glutDisplayFunc(render); // Enregistre le callback pour le redraw

        glewInit();

        return true;
    }

    void GlutEngine::Run() {
        glutTimerFunc(1000/SCREEN_FPS, mainloop,0);
        glutMainLoop();
    }

    // BasicAppInfo retourne une structure rempli des info par default
    APPINFO BasicAppInfo() {
        APPINFO info;
        strcpy(info.title,"Demo");
        info.windowHeight = DEFAULT_SCREEN_HEIGHT;
        info.windowWidth = DEFAULT_SCREEN_WIDTH;
        return info;
    }
};

#endif // __ENGINE_H__
