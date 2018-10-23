#include "../../include/engine.hpp"
#include "../../include/shaders.hpp"


using namespace ENGINE;

typedef std::vector<MyLine<float>> VLines; // Vecteur de lignes d'une dimension
typedef std::vector<std::vector<MyLine<float>>> V2Lines; // Vecteur de lignes de deux dimensions

enum MENU_OPTIONS { OPTION_ARBRE, OPTION_FLOCON , SAVE_DRAWING , EXIT_APP };


#define ROOT3 1.73205081
#define ONEBYROOT3 0.57735027

const float ONETHIRD = 1.0f/3.0f;
const float TWOTHIRD = 2.0f/3.0f;
const float vSin60 = glm::sin(glm::radians(60.0f));
const float vCos60 = glm::cos(glm::radians(60.0f));


float NumberGenerationTree = 4;


// DEFINITON VARIABLE GLOBALE

// Variable pour l'application glut
GlutEngine* app;

// Variable de l'identifiant de mon program de shader
unsigned int ShaderID;

std::vector<Position<float>> ListPointDraing;
int iterationNumber;
int drawingMode;

// DEFINITION FONCTION PRIVER

void keybinding(unsigned char key,int x,int y) {}
void specialkeybinding(int key, int x , int y) {}


const MyLine<float> startTriangleFlocon = { {-1.0,0.0},{1.0,0.0}};

VLines  LignesFlocon;
V2Lines LignesArbre;




std::vector<MyLine<float>> getFochLine(const MyLine<float> l) {
        /*float d = getLineLength(l.p1,l.p2);
        float offset = d/3.0f;
        float t = offset / d;
        Position<float> p2 = {(1.0f-t)*l.p1.x + t *l.p2.x, (1-t)*l.p1.y+ t * l.p2.y};
        t = offset*2.0f / d;
        Position<float> p4 = {(1.0f-t)*l.p1.x + t *l.p2.x, (1-t)*l.p1.y+ t * l.p2.y};
        // Calcul le p3 top du triangle
        float offsetx = offset/2.0f;
        float offsety = vSin60 * offset;
        Position<float> p3 = {p2.x+offsetx, p2.y+offsety};*/

        //float angle = 60.0f * glm::pi<float>()/180;

        Position<float> p2 = {(2.0f*l.p1.x+l.p2.x)/3.0f, (2.0f*l.p1.y+l.p2.y)/3.0f};
        Position<float> p4 = {(l.p1.x+2.0f*l.p2.x)/3.0f,(l.p1.y+2.0f*l.p2.y/3.0f)};

        float xx = p4.x - p2.x;
        float yy = p4.y - p2.y;
        Position<float> p3 = { xx * 0.5f + vSin60 * yy, xx * vSin60 + yy *0.5f };
        p3.x = p3.x+p2.x;
        p3.y = p3.y+p2.y;
        //Position<float> p3 = {(l.p1.x+l.p2.x)/2.0f - glm::sqrt(3.0f)/6.0f*(l.p2.y - l.p1.y), (l.p1.y+l.p2.y)/2.0f + glm::sqrt(3.0)/6.0 * (l.p2.x - l.p1.x)};
        
        
        //Position<float> p2 = {TWOTHIRD*l.p1.x + ONETHIRD*l.p2.x,TWOTHIRD*l.p1.y + ONETHIRD*l.p2.y};
        //Position<float> p3 = {0.5*(l.p1.x + l.p2.x) - 0.5*ONEBYROOT3*(l.p2.y - l.p1.y),0.5*(l.p1.y + l.p2.y) - 0.5*ONEBYROOT3*(l.p2.x - l.p1.x)};
        //Position<float> p4 = {ONETHIRD*l.p1.x + TWOTHIRD*l.p2.x,ONETHIRD*l.p1.y + TWOTHIRD*l.p2.y};
        return { {l.p1,p2},{p2,p3},{p3,p4},{p4,l.p2}};
        
}


// generateRecursiveTree
void generateRecursiveTree(int nbrGeneration, std::vector<std::vector<MyLine<float>>>& lines) {
    // Prend la derniere generation de branches (le derniere vecteur) pour en generer une nouvelle
    // si on n'est rendu a la generation 0 on quitte la boucle
    if(nbrGeneration == 0) return;

    std::cout << "Génération d'un arbre génération " << nbrGeneration << " il y a " << lines.size() << " generations dans la liste" << std::endl;

    std::vector<MyLine<float>> newGeneration;
    auto v = lines.back();
    for(auto line : v) {
        // Get la longeur de la ligne d'origine qu'on veut faire des childs
        float length = getLineLength(line);
        // Genere un angle aleatoire pour notre branche en 5 et 85 degrée
        float generateAngle = generateFloatInRange(5.0f,45.0f);
        // Genere la ratio a utiliser pour la longeur de la nouvelle branche
        float ratioPrevious = generateFloatInRange(0.3f,0.5f);

        // Get la longeur de notre nouveau segment
        float newLength = length * ratioPrevious;

        TrigoInfo<float> trigo = getTriangleInfoFromHypo(line,generateAngle,newLength);

        // Crée deux nouveau point mirroir pour les deux nouvelle branches avec les info de trigo obtenu
        Position<float> p1 = { line.p2.x+trigo.A, line.p2.y+trigo.H};
        Position<float> p2 = { line.p2.x-trigo.A, line.p2.y+trigo.H};

        // ajout les deux nouvelles lignes dans ma nouvelle generation
        newGeneration.push_back({line.p2,p1});
        newGeneration.push_back({line.p2,p2});
    }

    lines.push_back(newGeneration);
    
    generateRecursiveTree(--nbrGeneration,lines);
}


// Genere un nouvelle arbre dans la variable global Arbre
void generateNewTree() {
    LignesArbre.clear();
    // Génére la premiere génération de l'arbre au centre avec un tronc d'une longueur variable
    float length = generateFloatInRange(0.60f,1.15f);
    // Crée la ligne a partir du point de départ (0.0,-1.0)
    VLines start = { { {0.0f,-1.0f}, {0.0f,-1.0f+length} }};
    LignesArbre.push_back(start);
    generateRecursiveTree(NumberGenerationTree,LignesArbre);
}


void nextFoch() {
    std::vector<MyLine<float>> old;
    //ListLines.clear();
    for(auto l : old) {
        auto v = getFochLine(l);
        //ListLines.insert(ListLines.end(),v.begin(),v.end());
    }
}


// dessine le contenu d'un VLines
void drawVLines(const VLines& ListLines) {
    GLuint vboID;
    

    genBuffer(&vboID,0,ListLines.size() * sizeof(MyLine<float>),ListLines.data());
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE, 0,0);

    glDrawArrays(GL_LINES,0,ListLines.size()*2);

    glDisableVertexAttribArray(0);
}

// dessine le contenu d'un V2Lines;
void drawV2Lines(const V2Lines& ListLines) {
    GLuint vboID;

    for(auto l : ListLines) {
        int s = l.size();
        genBuffer(&vboID,0,l.size() * sizeof(MyLine<float>),l.data());
        glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE, 0,0);
        glEnableVertexAttribArray(0);
        glDrawArrays(GL_LINES,0,l.size()*2);
        glDisableVertexAttribArray(0);
    }
}




// Display mon quad dans une couleur qui change selon le keyboard
void display() {
    // clear le background avec la couleur voulu
    glClearColor(255,255,255,1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    // bind notre programme de shader
   	glUseProgram(ShaderID);

    glLineWidth(2.0f);
    glPointSize(5.0f);

    drawV2Lines(LignesArbre);

	glFlush();
}



void traitementMenuPrincipal(int value) {
    switch(value) {
        case OPTION_FLOCON: // Dessine un nouveau flocon

        break;
        case OPTION_ARBRE: // Dessine un nouvelle arbre

        break;
        case SAVE_DRAWING: // Sauvegarde l'oeuvre d'art vers un fait chier
            takeScreenShot();
        break;
        case EXIT_APP:
            glDeleteProgram(ShaderID);
            glutLeaveMainLoop();
        break;
    }
}

void createMenu() {
    int menuPrincipal;

    menuPrincipal = glutCreateMenu(traitementMenuPrincipal);
    glutAddSubMenu("Arbre Scholastique", OPTION_ARBRE);
    glutAddSubMenu("Flocon Kush", OPTION_FLOCON);
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


   generateNewTree();


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

   // Init du menu contextuel
   createMenu();

   app->Run();

   return 0;
}
