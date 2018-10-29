#ifndef __ENGINE_H__
#define __ENGINE_H__

#include "headers.hpp"

#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>



namespace ENGINE
{
    #define PI 3.14159265

    typedef float Mat4x4[4][4];

    const Mat4x4 MatriceTransformation { 
        { 1, 0, 0, 0},
        { 0, 1, 0, 0},
        { 0, 0, 1, 0},
        { 0, 0 ,0, 1}
     };

    // Constante de l'écran
    const int DEFAULT_SCREEN_WIDTH  = 800;
    const int DEFAULT_SCREEN_HEIGHT = 600;
    const int SCREEN_FPS    = 60;


    const float Maxndc = 1.0f;
    const float Minndc = -1.0f;


    struct Vecf {
        float x,y;
        Vecf(): x(0), y(0) {}
        Vecf(float a, float b) : x(a), y(b) {}

        Vecf operator+(const Vecf& p) {
            return Vecf(x+p.x,y+p.y);
        }
        Vecf operator-(const Vecf& p) {
            return Vecf(x-p.x,y-p.y);
        }


        void add(const Vecf& p) {
            x += p.x;
            y += p.y;
        }
        void sub(const Vecf& p) {
            x -= p.x;
            y -= p.y;
        }
        void scale(float n) {
            x = x*n;
            y = y*n;
        }

        float magnitude() {
            return sqrt(x*x+y*y);
        }

        void normalize() {
            float mag = magnitude();
            if(mag != 0) {
                scale(1/mag);
            }
        }

        Vecf rotate(float degrees) {
            double theta = (degrees * PI / 180.0f);
            double cosVal = cos(theta);
            double sinVal = sin(theta);
            double newX = x*cosVal - y*sinVal;
            double newY = x*sinVal + y*cosVal;
            return Vecf(newX,newY);
        }
    };


    template<typename T>
    struct Position {
        T x;
        T y;
    };



    template<typename T>
    struct MyLine {
        Position<T> p1;
        Position<T> p2;
        
    };

    struct VecLine {
        Position<float> p;
        Vecf v;
    };

    Position<float> getVecLineEndPoint(const VecLine& vl) {
        return { vl.p.x + vl.v.x, vl.p.y+vl.v.y};
    }



    template<typename T>
    struct TrigoInfo {
        T A; // Longeur de coté adjacent
        T H; // Longeur de l'hypothenuse
        T O; // Longeur du coté opposé

        T Angle; // Angle du triangle rectangle
    };

    float getLineLength(const Position<float>& p1,const Position<float>& p2) {
        return glm::sqrt(glm::pow(p2.x - p1.x,2)+glm::pow(p2.y - p1.y,2));
    }


    float getLineLength(const MyLine<float>& l) {
        return getLineLength(l.p1,l.p2);
    }

    // obtient les informations trigo d'un triangle rectangle formé avec l'angle données
    TrigoInfo<float> getTriangleInfoFromHypo(const MyLine<float>& l, float angle, float h) {
        TrigoInfo<float> t;
        t.Angle = 180-angle;
        t.H = h;
        float angleRadian = glm::radians(t.Angle);
        t.A = h * glm::sin(angleRadian);
        t.O = h * glm::cos(angleRadian);
        return t;
    }


    void Translation(Position<float> *points,int size,float valueX,float valueY) {
        for(int i=0;i<size;i++) {
            points[i].x += valueX;
            points[i].y += valueY;
        }
    }

    void Reflect(Position<float> *points,int size,bool abs,bool ord) {
        for(int i=0;i<size;i++) {
            if(abs)
                points[i].y = -points[i].y;
            if(ord)
                points[i].x = -points[i].x;
        }
    }


    


    template<typename T>
    struct RGBColor {
        T R;
        T G;
        T B;
    };

    
    float generateFloat() {
        return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

    std::mt19937 gen(std::random_device{}());

    float generateFloatInRange(float min, float max) {
        assert(max > min);
        std::uniform_real_distribution<float> dis(min,max);
        return dis(gen);
    }


    void renderString(float x,float y, void* font, const char* str) {
        glColor3f(0.0,0.0,0.0);
        glRasterPos2f(x,y);
        glutBitmapString(font,(unsigned char*)str);
    }

    void generateRandomColor(RGBColor<float>& randomColor) {
        randomColor.R = generateFloat(); randomColor.G = generateFloat(); randomColor.B = generateFloat();
    }

    // genBuffer generer un buffer
    void genBuffer(GLuint* id,int position,int size,const void * data) {
        glGenBuffers(1, id); // Generer le VBO
        glBindBuffer(GL_ARRAY_BUFFER,*id);  // Lier le VBO
        glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW); // Définir la taille, les données et le type du VBO
        glEnableVertexAttribArray(position); // Enable l'attribut qu'on veut activer
    }


    Position<float> ConvertToNDC(int x,int y) {
        Position<float> p;
        int sizeY = glutGet(GLUT_WINDOW_HEIGHT);
        int sizeX = glutGet(GLUT_WINDOW_WIDTH);
        p.x =  (x * (Maxndc - Minndc))/sizeX + Minndc;
        p.y = -(y-sizeY) * (Maxndc - Minndc)/sizeY + Minndc;
        return p;
    }

    void takeScreenShot() {
        int gW = glutGet(GLUT_WINDOW_WIDTH);
        int gH = glutGet(GLUT_WINDOW_HEIGHT);

        unsigned char* buffer = (unsigned char*)malloc(gW * gH * 3);
        glReadPixels(0,0,gW,gH,GL_RGB,GL_UNSIGNED_BYTE,buffer);
        char name[512];
        long int t = time (NULL);
        sprintf(name,"screenshot_%ld.png",t);


        unsigned char* last_row = buffer + (gW * 3 * (gH - 1));
        if(!stbi_write_png(name,gW,gH,3,last_row, -3 * gW)) {
            std::cerr << "Error: could not write screenshot file " << name << std::endl;
        }
    }



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
        bool    mode3d;


        void(*render)(void) = NULL;
        void(*mainloop)(int) = NULL;
        void(*keybinding)(unsigned char,int,int) = NULL;
        void(*funckeybinding)(int,int,int) = NULL;
        void(*mousebinding)(int,int,int,int) = NULL;
        void(*mouseMove)(int,int) = NULL;


        public:
            GlutEngine(int) {
                isMax = false;
            }
            GlutEngine(bool v) {
                mode3d = v;

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
            void SetMouseFunc(void(*m)(int,int,int,int)) { mousebinding = m; }
            void SetMouseMouveFunc(void(*m)(int,int)) { mouseMove = m; }
      
            void EndApp();

        private:
            bool InitGL();
    
    };


    bool GlutEngine::InitGL() {
        // Initialize la matrix de projection
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        // Initialize le mode Ortho pour pour changer le system de cordoner
        //glOrtho(0.0,wWidth,wHeight,0.0,1.0,-1.0);

        glPolygonMode(GL_FRONT,GL_FILL);

        // Initialize la matrix de modelview
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // Initialize la couleur pour clean l'ecran
        glClearColor(0.f,0.f,0.f,1.f);

        if(mode3d) {
            glEnable(GL_DEPTH_TEST);
        }

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

        srand (static_cast <unsigned> (time(0)));
        // Crée une windows double buffer le gros
        if(mode3d) {
            glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);

        } else {
            glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);
        }

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
        glutMouseFunc(mousebinding);
        glutMotionFunc(mouseMove);
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
