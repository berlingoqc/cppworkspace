#include "../../include/engine.hpp"
#include "../../include/shaders.hpp"
using namespace ENGINE;

GlutEngine* app;


unsigned int ShaderID;
unsigned int vaoID[1];


void createDrawingLine() {
    /*
        1. Generer le vertex array object
        2. Lier le vertex array object
        3. Générer le vertex buffer object
        4. Lier le vertex buffer object
    */
    GLuint vboID;
    
    GLfloat sommets[4] = {
        -1.0f,-1.0f,1.0f,1.0f
    };

    //glGenVertexArrays(1, &vaoID[0]); // Crée le VAO
    //glBindVertexArray(vaoID[0]); // Lier le VAO pour l'utiliser


    glGenBuffers(1, &vboID); // Generer le VBO
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER,vboID);  // Lier le VBO
    glBufferData(GL_ARRAY_BUFFER, 4*sizeof(GLfloat), sommets, GL_STATIC_DRAW); // Définir la taille, les données et le type du VBO
    
    glVertexAttribPointer(0,2, GL_FLOAT, GL_FALSE, 0,0); // Définit le pointeur d'attributs des sommets

    //glEnableVertexAttribArray(0); // Désactive le VAO
    //glBindVertexArray(0); // Désactiver le VBO
    /*
	GLuint buffSommets, buffCouleurs;

	GLfloat sommets[20] = {
		-0.5f, 0.0f, 0.0f,1.0f,
		0.5f, 0.0f, 0.0f,1.0f,
		-0.25f, -1.0f, 0.0f,1.0f,
		0.0f, 0.5f, 0.0f, 1.0f,
		0.25f, -1.0f, 0.0f,1.0f
	};

	glGenBuffers(1, &buffSommets);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, buffSommets);
	glBufferData(GL_ARRAY_BUFFER, sizeof(sommets), sommets, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    */
}


// Display mon quad dans une couleur qui change selon le keyboard
void display() {
    glClearColor(0.2f,0.3f,0.3f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

   	glUseProgram(ShaderID);
	

    createDrawingLine();
	//glLineWidth(15.0);
	glDrawArrays(GL_LINES, 0, 2);

	glDisableVertexAttribArray(0);
	//glDisableVertexAttribArray(1);

	glFlush();
}

void keybinding(unsigned char key,int x,int y) {

}
void specialkeybinding(int key, int x , int y) {

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
   app->Init(info,argc,argv);


   // Initialize no MyShaders
   ENGINE::MyShader MyShader;
   if(!MyShader.OpenMyShader("vertex.glsl","fragment.glsl")) {
       return -1;
   }
   ShaderID = MyShader.GetShaderID();

   app->Run();
   return 0;
}
