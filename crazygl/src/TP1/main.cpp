#include "../../include/engine.hpp"
#include "../../include/shaders.hpp"


using namespace ENGINE;

// DEFINITION ENUM
enum MENU_OPTIONS { CLEAR_DRAWING, DRAWING_MODE, COLOR_MODE, BACKGROUND_RESET,SAVE_DRAWING, EXIT_APP };
enum DRAWING_OPTIONS { DRAW_POINTS, DRAW_LINES, DRAW_TRIANGLE, DRAW_QUADS, DRAW_FREE };
enum COLORS_OPTIONS { RED_OPTION=0, GREEN_OPTION, BLUE_OPTION, WHITE_OPTION, RANDOM_OPTION };

const RGBColor<float> colors[4] {
    { 1.0, 0.0, 0.0 }, // RED
    { 0.0, 1.0, 0.0 }, // GREEN
    { 0.0, 0.0, 1.0 }, // BLUE
    { 1.0, 1.0, 1.0 }  // WHITE
};

const std::string shapesName[5] {
    "Points", "Lignes", "Triangles", "Quads", "Ligne continue"
};
const std::string colorsName[5] {
    "Rouge", "Vert", "Blue", "Blanc", "Aleatoire"
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

RGBColor<float> randomColor;
RGBColor<float> randomColorBG;

bool leftPressed = false;

GLenum ShapeMode = GL_POINTS;
int NumberSommetShape = 1;

std::vector<Position<float>> ListPointDrawing;
std::vector<RGBColor<float>> ListColorDrawing;
std::vector<Position<float>> ListPointEnter;

// DEFINITION FONCTION PRIVER

void keybinding(unsigned char key,int x,int y) {}
void specialkeybinding(int key, int x , int y) {}


// createVBOVector crée le vbo avec le vecteur de point a dessiner
void createVBOVector() {
    GLuint vboID;

    genBuffer(&vboID,0,ListPointDrawing.size() * sizeof(Position<float>),ListPointDrawing.data());
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE, 0,0);
}

// createVBOColor crée le vbo pour les couleurs des vertexs
void createVBOColor() {
    GLuint vboID;

    genBuffer(&vboID,1,ListColorDrawing.size() * sizeof(RGBColor<float>),ListColorDrawing.data());
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,0,0);
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
        glClearColor(randomColorBG.R,randomColorBG.G,randomColorBG.B,1.0f);
    } else {
        glClearColor(colors[ColorBackGround].R,colors[ColorBackGround].G,colors[ColorBackGround].B,1.0f);
    }
    glClear(GL_COLOR_BUFFER_BIT);
}


// Display mon quad dans une couleur qui change selon le keyboard
void display() {
    // clear le background avec la couleur voulu
    setBackGroundColor();
    // bind notre programme de shader
   	glUseProgram(ShaderID);

    glPointSize(10.0);
    if(DrawingMode == DRAW_FREE) {
        glLineWidth(3.0);
    } else {
        glLineWidth(10.0);
    }


    // Dessine les formes entrer
    createVBOVector();
    createVBOColor();
    glDrawArrays(ShapeMode,0,ListPointDrawing.size());

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    // Dessine les points entrer pour la prochaine former a dessiner
    createVBOPoint();
    glDrawArrays(GL_POINTS,0,ListPointEnter.size());


	glDisableVertexAttribArray(0);

    std::stringstream info;
    info << "FORME : " << shapesName[DrawingMode] << "      COULEUR : " << colorsName[ColorDrawing] << "    " << ListPointEnter.size() << "/"<< NumberSommetShape;
    renderString(-1.0,0.95,GLUT_BITMAP_TIMES_ROMAN_10,info.str().c_str());

	glFlush();
}



void onNewPosition(Position<float> p) {
        // Ajout le point a la liste temporaire de point
        ListPointEnter.push_back(p);
        // Si on n'a le nombre de sommet pour la shape voulue
        if(ListPointEnter.size() == NumberSommetShape) {
            // Pour le nombre de point ajout la couleur
        
            for(int i=0; i<ListPointEnter.size();i++) {
                RGBColor<float> f;
                if(ColorDrawing == RANDOM_OPTION) {
                    generateRandomColor(f);
                } else {
                    f = colors[ColorDrawing];
                }
                ListColorDrawing.push_back(f);
            }
            // Envoie le contenu dans ma list de point a dessiner et vide le contenu de cette forme
            std::copy(ListPointEnter.begin(),ListPointEnter.end(),std::back_inserter(ListPointDrawing));

            ListPointEnter.clear();
        } 
} 

void mouseMouve(int x, int y) {
    if(leftPressed && DrawingMode == DRAW_FREE) {
        Position<float> p = ENGINE::ConvertToNDC(x,y);
        ListPointDrawing.push_back(p);
        RGBColor<float> f;
        if(ColorDrawing == RANDOM_OPTION) {
            generateRandomColor(f);
        } else {
            f = colors[ColorDrawing];
        }
        ListColorDrawing.push_back(f);
     }
}

void mousebinding(int button, int state, int x,int y) {
    Position<float> p = ENGINE::ConvertToNDC(x,y);
    if(button == 0 && state == 0) {
        leftPressed = true;
        if(DrawingMode == DRAW_FREE) return;
        onNewPosition(p);
    } else if (button == 0 && state == 1) {
        leftPressed = false;
    }
}


void traitementMenuPrincipal(int value) {
    switch(value) {
        case CLEAR_DRAWING:
            // Clear les vectors de point
            ListPointDrawing.clear();
            ListColorDrawing.clear();
            ListPointEnter.clear();
        break;
        case BACKGROUND_RESET:
            // Set la couleur du background avec la couleurs selectioner pour dessiner
            ColorBackGround = ColorDrawing;
            if(ColorDrawing == RANDOM_OPTION) {
                generateRandomColor(randomColorBG);
            }
        break;
        case SAVE_DRAWING:
            // Démare l'operation de sauvegarde dans un thread pour pas freeze le display
            takeScreenShot();
        break;
        case EXIT_APP:
            glDeleteProgram(ShaderID);
            glutLeaveMainLoop();
        break;
    }
}

void traitementMenuCouleur(int value) {
    ColorDrawing = (COLORS_OPTIONS)value;
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
        case DRAW_FREE:
            DrawingMode = DRAW_FREE;
            NumberSommetShape = 0;
            ShapeMode = GL_LINE_STRIP;
        break;
    }

    ListPointEnter.clear();
    ListColorDrawing.clear();
    ListPointDrawing.clear();
}

void createMenu() {
    int menuPrincipal, smShape, smCouleur;

    smShape = glutCreateMenu(traitementMenuShape);
    glutAddMenuEntry("Point", DRAW_POINTS);
    glutAddMenuEntry("Ligne", DRAW_LINES);
    glutAddMenuEntry("Triangle", DRAW_TRIANGLE);
    glutAddMenuEntry("Quad", DRAW_QUADS);
    glutAddMenuEntry("Ligne continue",DRAW_FREE);

    smCouleur = glutCreateMenu(traitementMenuCouleur);
    glutAddMenuEntry("Rouge", RED_OPTION);
    glutAddMenuEntry("Bleue", BLUE_OPTION);
    glutAddMenuEntry("Vert", GREEN_OPTION);
    glutAddMenuEntry("Blanc", WHITE_OPTION);
    glutAddMenuEntry("Aleatoire", RANDOM_OPTION);

    menuPrincipal = glutCreateMenu(traitementMenuPrincipal);
    glutAddSubMenu("Couleurs",smCouleur);
    glutAddSubMenu("Formes", smShape);
    glutAddMenuEntry("Reset BG", BACKGROUND_RESET);
    glutAddMenuEntry("Effacer", CLEAR_DRAWING);
    glutAddMenuEntry("Sauvegarder", SAVE_DRAWING);
    glutAddMenuEntry("Exit",EXIT_APP);

    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void mainLoop(int val) {
    glutPostRedisplay();
    // Roule la frame une autre fois
    //mainLoop(val);
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
   app->SetMouseMouveFunc(mouseMouve);
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
