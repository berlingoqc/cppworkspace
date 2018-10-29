#include "../../include/engine.hpp"
#include "../../include/shaders.hpp"
using namespace ENGINE;

// Variable globale pour mon engine (wrapper autour de glut)
GlutEngine* app;

// Variable pour l'identifiant de mon program de shader compiler
unsigned int ShaderID;

// Scale utiliser comme variable uniform pour le dessin
float Scale = 1.0f;

// Vector content la liste de point utilisé pour render les formes a l'écran
std::vector<Position<float>> listPoint;

enum OptionMenu { SCALE_100, SCALE_60, EXIT_APP };


void createDrawing() {
    GLfloat vertices[24] {
        -1.0f, 0.0f, // Premiere droite
        -0.75f, 0.5f,
        
        -0.75f, 0.5f, // Deuxieme droite
        -0.25, 0.95f,

        -0.25f, 0.95f, // Troisieme droite
        0.25f, 0.95f,

        0.25f, 0.95f, // Quatrieme droite
        0.75f, 0.5f,

        0.75f, 0.5f, // Cinquieme droite
        1.0f, 0.0f,

        -1.0f,0.0f, // Base
        1.0f, 0.0f
    };
    GLuint vbo;
    genBuffer(&vbo,0,sizeof(vertices),vertices);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,0,0);
}

void createDrawing2() {
    GLfloat vertices[8] {
        0.0f, 0.0f,
        0.0f, -0.95f,
        0.2f, -0.95f,
        0.2f, -0.90f
    };
    GLuint vbo;
    genBuffer(&vbo,0,sizeof(vertices),vertices);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,0,0);
}

// Display mon quad dans une couleur qui change selon le keyboard
void display() {
    glClearColor(0.2f,0.3f,0.3f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

   	glUseProgram(ShaderID);

    // Change la valeur de scale 
    GLint loc = glGetUniformLocation(ShaderID,"Scale");
    if (loc != -1) {
        glUniform1f(loc, Scale);
    }

    createDrawing();
	glDrawArrays(GL_POLYGON,0,12);

    createDrawing2();
    glLineWidth(5.0f);
    glDrawArrays(GL_LINE_STRIP,0,4);

	
    glDisableVertexAttribArray(0);

	glFlush();
}

void mousebinding(int button, int state, int x,int y) {
    Position<float> p = ENGINE::ConvertToNDC(x,y);
    if(button == 0 && state == 0) {
        listPoint.push_back(p); 
    }
}

void keybinding(unsigned char key,int x,int y) {

}
void specialkeybinding(int key, int x , int y) {

}

void traitementMenu(int value) {
    switch(value) {
        case SCALE_100:
            Scale = 1.0f;
        break;
        case SCALE_60:
            Scale = 0.6f;
        break;
        case EXIT_APP:
            glDeleteProgram(ShaderID);
            glutLeaveMainLoop();
        break;
    }
}

void createMenu() {
    int menu = glutCreateMenu(traitementMenu);
    glutAddMenuEntry("Scale 100",0);
    glutAddMenuEntry("Scale 60", 1);
    glutAddMenuEntry("Quitter",EXIT_APP);

    glutAttachMenu(GLUT_RIGHT_BUTTON);

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
   app->SetMouseFunc(mousebinding);
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
