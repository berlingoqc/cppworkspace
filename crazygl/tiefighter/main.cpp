#include "engine.h"
#include "shaders.h"
#include "city.h"
#include "camera.h"

//#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"


using namespace ENGINE;

// DEFINITON VARIABLE GLOBALE
typedef unsigned int uint;


ProceduralCity	city;


static void glfw_error_callback(int error,const char* description)
{
	std::cerr << "GLFW Error " << error << " : " << description << std::endl;
}

int main(int argc, char** argv) {
	glfwSetErrorCallback(glfw_error_callback);
	if(!glfwInit())
	{
		std::cerr << "Erreur initialisation glfw" << std::endl;
		return 1;
	}

	GLFWwindow*	window = glfwCreateWindow(1600, 900, "TP Ville Procedurale", nullptr,nullptr);
	if(!window)
	{
		glfwTerminate();
		std::cerr << "Erreur creation de la fenetre glfw" << std::endl;
		return 1;
	}
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1); // vsync

	int v = glewInit();

	float z = 1.0f;

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls

	// Setup Platform/Renderer bindings
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");

	// Setup Style
	ImGui::StyleColorsDark();


	bool show_demo_window = true;
	bool show_another_window = false;
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
	//glutSetCursor(GLUT_CURSOR_NONE);
	glfwSetInputMode(window, GLFW_CURSOR_DISABLED, 0);

	if(!city.configure("textures"))
	{
		return 1;
	}

	city.load();

	while(!glfwWindowShouldClose(window))
	{


		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		city.render();

		// Rendering
		ImGui::Render();
		int display_w, display_h;
		glfwMakeContextCurrent(window);
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


		glfwMakeContextCurrent(window);

		glfwSwapBuffers(window);

		glfwPollEvents();
	}

	glfwTerminate();


	return 0;
}

