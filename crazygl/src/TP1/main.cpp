#include "../../include/engine.hpp"
#include "../../include/shaders.hpp"

using namespace ENGINE;

// DEFINITION ENUM
enum MENU_OPTIONS { CLEAR_DRAWING, DRAWING_MODE, COLOR_MODE, BACKGROUND_RESET, EXIT_APP };
enum DRAWING_OPTIONS { DRAW_POINTS, DRAW_LINES, DRAW_TRIANGLE, DRAW_QUADS };
enum COLORS_OPTIONS { RED_OPTION=0, GREEN_OPTION, BLUE_OPTION, WHITE_OPTION, RANDOM_OPTION };

// Définition de mes couleurs
const GLfloat colors [4][3] {
    { 1.0, 0.0, 0.0 }, // RED
    { 0.0, 1.0, 0.0 }, // GREEN
    { 0.0, 0.0, 1.0 }, // BLUE
    { 1.0, 1.0, 1.0 }  // WHITE
};

// DEFINITON VARIABLE GLOBALE

// Variable pour l'application glut
GlutEngine* app;


// Variable de l'identifiant de mon program de shader
unsigned int ShaderID;

// Variable du mode de drawing courrant
DRAWING_OPTIONS DrawingMode = DRAW_POINTS;
// Variable de la couleur utilisé pour clear le fond ecran ou pour draw
COLORS_OPTIONS  ColorDrawing = RED_OPTION;
COLORS_OPTIONS  ColorBackGround = WHITE_OPTION;

GLfloat randomColor[3];

GLenum ShapeMode = GL_POINTS;
int NumberSommetShape = 1;

std::vector<Position<float>> ListPointDrawing;
std::vector<Position<float>> ListPointEnter;

// DEFINITION FONCTION PRIVER

void keybinding(unsigned char key,int x,int y) {}
void specialkeybinding(int key, int x , int y) {}


void generateRandomColor() {
    randomColor[0] = generateFloat(); randomColor[1] = generateFloat(); randomColor[3] = generateFloat();
}

// genBuffer generer un buffer
void genBuffer(GLuint* id,int position,int size,const void * data) {
    glGenBuffers(1, id); // Generer le VBO
    glBindBuffer(GL_ARRAY_BUFFER,*id);  // Lier le VBO
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW); // Définir la taille, les données et le type du VBO
    glEnableVertexAttribArray(position); // Enable l'attribut qu'on veut activer
}

// createVBOVector crée le vbo avec le vecteur de point a dessiner
void createVBOVector() {
    GLuint vboID;

    genBuffer(&vboID,0,ListPointDrawing.size() * sizeof(Position<float>),ListPointDrawing.data());
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE, 0,0);
}

// createVBOPoint crée le vbo pour les points inserer pour la forme
void createVBOPoint() {
    GLuint vboID;

    genBuffer(&vboID,0,ListPointEnter.size() * sizeof(Position<float>),ListPointEnter.data());
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE, 0,0);
}

// setBackgroundColor set la background color selon le mode courrant
void setBackGroundColor() {
    if(ColorBackGround == RANDOM_OPTION) {
        glClearColor(randomColor[0],randomColor[1],randomColor[2],1.0f);
    } else {
        glClearColor(colors[ColorBackGround][0],colors[ColorBackGround][1],colors[ColorBackGround][2],1.0f);
    }
    glClear(GL_COLOR_BUFFER_BIT);
}


// Display mon quad dans une couleur qui change selon le keyboard
void display() {
    // clear le background avec la couleur voulu
    setBackGroundColor();
    // bind notre programme de shader
   	glUseProgram(ShaderID);



    // Dessine les formes entrer
    createVBOVector();
    glDrawArrays(ShapeMode,0,ListPointDrawing.size());


    // Dessine les points entrer pour la prochaine former a dessiner
    glPointSize(10.0);
    createVBOPoint();
    glDrawArrays(GL_POINTS,0,ListPointEnter.size());


	glDisableVertexAttribArray(0);


	glFlush();
}

void mousebinding(int button, int state, int x,int y) {
    Position<float> p = ENGINE::ConvertToNDC(x,y);
    if(button == 0 && state == 0) {
        // Ajout le point a la liste temporaire de point
        ListPointEnter.push_back(p);
        // Si on n'a le nombre de sommet pour la shape voulue
        if(ListPointEnter.size() == NumberSommetShape) {
            // Envoie le contenu dans ma list de point a dessiner et vide le contenu de cette forme
            std::copy(ListPointEnter.begin(),ListPointEnter.end(),std::back_inserter(ListPointDrawing));
            ListPointEnter.clear();
        }
    }
}
void traitementMenuPrincipal(int value) {
    switch(value) {
        case CLEAR_DRAWING:
            // Clear les vectors de point
            ListPointDrawing.clear();
            ListPointEnter.clear();
        break;
        case BACKGROUND_RESET:
            // Set la couleur du background avec la couleurs selectioner pour dessiner
            ColorBackGround = ColorDrawing;
            if(ColorDrawing == RANDOM_OPTION) {
                generateRandomColor();
            }
        break;
        case EXIT_APP:
            glDeleteProgram(ShaderID);
            glutLeaveMainLoop();
        break;
    }
}

void traitementMenuCouleur(int value) {
    ColorDrawing = (COLORS_OPTIONS)value;
    if (value == RANDOM_OPTION) {
        
    }
}

void traitementMenuShape(int value) {
    switch(value) {
        case DRAW_POINTS:
            DrawingMode = DRAW_POINTS;
            NumberSommetShape = 1;
            ShapeMode = GL_POINTS;
        break;
        case DRAW_LINES:
            DrawingMode = DRAW_LINES;
            NumberSommetShape = 2;
            ShapeMode = GL_LINES;
        break;
        case DRAW_TRIANGLE:
            DrawingMode = DRAW_TRIANGLE;
            NumberSommetShape = 3;
            ShapeMode = GL_TRIANGLES;
        break;
        case DRAW_QUADS:
            DrawingMode = DRAW_QUADS;
            NumberSommetShape = 4;
            ShapeMode = GL_QUADS;
        break;
    }

    ListPointEnter.clear();
    ListPointDrawing.clear();
}

void createMenu() {
    int menuPrincipal, smShape, smCouleur;

    smShape = glutCreateMenu(traitementMenuShape);
    glutAddMenuEntry("Point", DRAW_POINTS);
    glutAddMenuEntry("Ligne", DRAW_LINES);
    glutAddMenuEntry("Triangle", DRAW_TRIANGLE);
    glutAddMenuEntry("Quad", DRAW_QUADS);

    smCouleur = glutCreateMenu(traitementMenuCouleur);
    glutAddMenuEntry("Rouge", RED_OPTION);
    glutAddMenuEntry("Bleue", BLUE_OPTION);
    glutAddMenuEntry("Vert", GREEN_OPTION);
    glutAddMenuEntry("Blanc", WHITE_OPTION);
    glutAddMenuEntry("Aléatoire", RANDOM_OPTION);

    menuPrincipal = glutCreateMenu(traitementMenuPrincipal);
    glutAddSubMenu("Couleurs",smCouleur);
    glutAddSubMenu("Formes", smShape);
    glutAddMenuEntry("Reset BG", BACKGROUND_RESET);
    glutAddMenuEntry("Effacer", CLEAR_DRAWING);
    glutAddMenuEntry("Exit",EXIT_APP);

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
