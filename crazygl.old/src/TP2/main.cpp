#include "../../include/engine.hpp"
#include "../../include/shaders.hpp"


using namespace ENGINE;

typedef std::vector<MyLine<float>> VLines; // Vecteur de lignes d'une dimension
typedef std::vector<std::vector<MyLine<float>>> V2Lines; // Vecteur de lignes de deux dimensions

enum MENU_OPTIONS { OPTION_IDLE ,OPTION_ARBRE, OPTION_FLOCON , OPTION_SETTINGS, SAVE_DRAWING ,
    OPTION_NBR_ITERATION, EXIT_APP };


const float vSin60 = glm::sin(glm::radians(60.0f));
const float vCos60 = glm::cos(glm::radians(60.0f));

// DEFINITON VARIABLE GLOBALE

// Variable pour l'application glut
GlutEngine* app;

// Variable de l'identifiant de mon program de shader
unsigned int ShaderID;

// Nombre d'itération de la fonction recursive dans la génération d'une forme
int iterationNumber = 0;

// Mode de dessin actuel (arbre, flaucon ou rien)
int drawingMode = OPTION_IDLE;;


const std::vector<MyLine<float>> startTriangleFlocon = { 
    {{-0.5,0.5}, {0.5,0.5}}, 
    {{-0.5,0.5}, {0.0,-0.5}},
    {{0.5, 0.5}, {0.0,-0.5}}
};

VLines  LignesFlocon;
V2Lines LignesArbre;


void generateRecursiveFoch(int nbrIteration, std::vector<MyLine<float>> lines) {
    if(nbrIteration == 0) {
        LignesFlocon = lines;
        return;
    }

    std::cout << "Nouvelle iteration Foch : il y a " << lines.size() << " lignes" << std::endl;

    // Crée un nouveau vecteur pour nos nouvelles lignes de la grandeur de 4 fois le nombre de lignes passé
    std::vector<MyLine<float>> newLine(lines.size()*4);

    for(auto line : lines) {
        // Va chercher le point 2 au 1/3 et le point 4 au 2/3 du segment
        Position<float> p2 = { (line.p1.x + (line.p2.x - line.p1.x) / 3.0f), (line.p1.y + (line.p2.y - line.p1.y) / 3.0f ) };
        Position<float> p4 = { (line.p1.x + 2.0f * (line.p2.x - line.p1.x) / 3.0f ), (line.p1.y + 2.0f * (line.p2.y - line.p1.y) / 3.0f) };
        // Va chercher le point du sommet de notre triangle
        Position<float> p3 = {((p2.x+p4.x) * vCos60 - (p4.y-p2.y) * vSin60), ((p2.y+p4.y) * vCos60 + (p4.x-p2.x) * vSin60)};

        // Ajout les 4 nouvelles lignes dans notre vecteur
        newLine.push_back({line.p1,p2});
        newLine.push_back({p2,p3});
        newLine.push_back({p3,p4});
        newLine.push_back({p4,line.p2});
    }
    generateRecursiveFoch(--nbrIteration,newLine);
}

void generateNewFoch() {
    generateRecursiveFoch(iterationNumber,startTriangleFlocon);
}


// generateRecursiveTree
void generateRecursiveTree(int nbrGeneration, std::vector<std::vector<MyLine<float>>>& lines) {
    // Prend la derniere generation de branches (le derniere vecteur) pour en generer une nouvelle
    // si on n'est rendu a la generation 0 on quitte la boucle
    if(nbrGeneration == 0) return;

    std::cout << "Generation " << nbrGeneration << " il y a " << lines.size() << " generations dans la liste" << std::endl;

    std::vector<MyLine<float>> newGeneration;
    auto v = lines.back();
    for(auto line : v) {
        // Get la longeur de la ligne d'origine qu'on veut faire des childs
        float length = getLineLength(line);
        // Genere un angle aleatoire pour notre branche en 5 et 85 degrée
        float generateAngle = generateFloatInRange(5.0f,85.0f);
        // Genere la ratio a utiliser pour la longeur de la nouvelle branche
        float ratioPrevious = generateFloatInRange(0.45f,0.8f);

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
    float length = generateFloatInRange(0.30f,0.9f);
    // Crée la ligne a partir du point de départ (0.0,-1.0)
    VLines start = { { {0.0f,-1.0f}, {0.0f,-1.0f+length} }};
    LignesArbre.push_back(start);
    generateRecursiveTree(iterationNumber,LignesArbre);
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

    glLineWidth(3.0f);

    if(drawingMode == OPTION_ARBRE) {
        drawV2Lines(LignesArbre);
    } else if (drawingMode == OPTION_FLOCON) {
        drawVLines(LignesFlocon);
    }

    if(drawingMode != OPTION_IDLE) {
        std::stringstream info;
        info << "Nombre de generation : " << iterationNumber;
        renderString(-1.0,0.95,GLUT_BITMAP_TIMES_ROMAN_10,info.str().c_str());
    }

	glFlush();
}


void traitementMenuSettings(int value) {
    if(value < 0 && iterationNumber > 0) {
        iterationNumber--;
    } else if (value > 0) {
        iterationNumber++;
    }
    if(drawingMode == OPTION_FLOCON) {
        generateNewFoch();
    } else if (drawingMode == OPTION_ARBRE) {
        generateNewTree();
    }
}

void traitementMenuPrincipal(int value) {
    switch(value) {
        case OPTION_FLOCON: // Dessine un nouveau flocon
            drawingMode = OPTION_FLOCON;
            generateNewFoch();
        break;
        case OPTION_ARBRE: // Dessine un nouvelle arbre
            drawingMode = OPTION_ARBRE;
            generateNewTree();
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
    int menuPrincipal, menuOption;

    menuOption = glutCreateMenu(traitementMenuSettings);
    glutAddMenuEntry("iteration++",1);
    glutAddMenuEntry("iteration--",-1);

    menuPrincipal = glutCreateMenu(traitementMenuPrincipal);
    glutAddMenuEntry("Arbre Scholastique", OPTION_ARBRE);
    glutAddMenuEntry("Flocon Kush", OPTION_FLOCON);
    glutAddSubMenu("Options",menuOption);
    glutAddMenuEntry("Sauvegarder", SAVE_DRAWING);
    glutAddMenuEntry("Exit",EXIT_APP);

    glutAttachMenu(GLUT_RIGHT_BUTTON);
}


void keybinding(unsigned char key,int x,int y) {
    switch(key) {
        case 27: // escape key
            glutLeaveMainLoop(); // Fin du programme
        break;
        case 'a' | 'A':
            traitementMenuPrincipal(OPTION_ARBRE);
        break;
        case 'k' | 'K':
            traitementMenuPrincipal(OPTION_FLOCON);
        break;
    }
}


void mainLoop(int val) {
    glutPostRedisplay();
    glutTimerFunc(1000/SCREEN_FPS,mainLoop,val);
}

int main(int argc,char** argv) {

   ENGINE::APPINFO info = ENGINE::BasicAppInfo();
   GlutEngine g(0);
   app = &g;
   app->SetMainFunc(mainLoop);
   app->SetRenderFunc(display);
   app->SetKeyFunc(keybinding);
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
