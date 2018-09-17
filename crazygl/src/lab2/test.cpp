#include "../../include/engine.hpp"
#include "../../include/shaders.hpp"
using namespace ENGINE;

GlutEngine* app;


unsigned int ShaderID;

void createDrawing()
{
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

	GLfloat couleurs[20] = {
		0.5f, 0.0f, 0.0f,1.0f,
		0.5f, 0.0f, 0.0f,1.0f,
		0.0f, 1.0f, 0.0f,1.0f,
		0.0f, 0.0f, 1.0f, 1.0f,
		0.0f, 1.0f, 0.3f,1.0f
	};

	glGenBuffers(1, &buffCouleurs);
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, buffCouleurs);
	glBufferData(GL_ARRAY_BUFFER, sizeof(couleurs), couleurs, GL_STATIC_DRAW);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
	
}

void createDrawingLine() {
    GLuint buffSommets;
    GLfloat sommets[4] = {
        -1.0f,-1.0f,1.0f,1.0f
    };
    glGenBuffers(1, &buffSommets);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER,buffSommets);
    glBufferData(GL_ARRAY_BUFFER, sizeof(sommets), sommets, GL_STATIC_DRAW);
    glVertexAttribPointer(0,4, GL_FLOAT, GL_FALSE, 0,0);
}


// Display mon quad dans une couleur qui change selon le keyboard
void display() {
    glClearColor(0.2f,0.3f,0.3f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    glUseProgram(ShaderID);

    createDrawingLine();

    glDrawArrays(GL_LINE,0,1);
    glDisableVertexAttribArray(0);


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
