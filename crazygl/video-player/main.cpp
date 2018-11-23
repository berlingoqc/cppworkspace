#include <iostream>

#include <GL\glew.h>
#include <GLFW/glfw3.h>

#include <opencv2\core.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\videoio.hpp>

#include "rgb_player.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"


// DEFINITON VARIABLE GLOBALE
typedef unsigned int uint;

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

	cv::VideoCapture vc("main.mp4");
	if (!vc.isOpened()) {
		return 1;
	}
	YUV420P_Player player;
	if (!player.setup(640, 360)) {
		printf("FAILED\n");
	}

	std::vector<cv::Mat> channels(3);
	cv::Mat m;

	while(!glfwWindowShouldClose(window))
	{


		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		{
			static int range_building[1] {0};
			ImGui::Begin("Configuration Ville Procedural");

			ImGui::TextColored({ 255,255,0,1 }, "Configuration generation batiment");
			ImGui::SliderInt("Max", &range_building[0], 0, 400);


			if (ImGui::Button("Regenerer")) {

			}

			ImGui::End();
		}


		if (!vc.read(m)) {
			glfwTerminate();
			return 1;
		}
		//cv::cvtColor(m, m, cv::COLOR_BGR2YUV);
		cv::resize(m, m, cv::Size(640, 360));

		//cv::split(m, channels);

		//player.setYPixels(channels[0].ptr<uint8_t>(), channels[0].step);
		//player.setUPixels(channels[1].ptr<uint8_t>(), channels[1].step);
		//player.setVPixels(channels[2].ptr<uint8_t>(), channels[2].step);

		player.setPixels(m);
		player.draw(100, 100);

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

