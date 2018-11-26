#include "camera.h"


#include <iostream>
#include "imgui.h"
#include <GLFW/glfw3.h>

FPSCamera::FPSCamera()
{
}

glm::mat4 FPSCamera::getLookAt()
{
	return glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);
}

glm::mat4 FPSCamera::getView()
{
	return 	glm::mat4(glm::mat3(glm::lookAt(camera_pos, camera_pos + camera_front, camera_up)));
}

void FPSCamera::update()
{
	GLfloat	cameraSpeed = 0.8f;
	glm::vec3 camDevant;
	camDevant.x = camera_front.x;
	camDevant.z = camera_front.z;
	camDevant.y = 0.0f;

	ImGuiIO& io = ImGui::GetIO();

	if (io.KeysDown[GLFW_KEY_LEFT_SHIFT])
		mouse_move(0, 0);
	if (io.KeysDown[GLFW_KEY_A]) {
		camera_pos -= glm::normalize(glm::cross(camDevant, camera_up)) * cameraSpeed;
	}
	if (io.KeysDown[GLFW_KEY_W]) {
		camera_pos += cameraSpeed * camDevant;
	}
	if (io.KeysDown[GLFW_KEY_S]) {
		camera_pos -= cameraSpeed * camDevant;
	}
	if (io.KeysDown[GLFW_KEY_D]) {
		camera_pos += glm::normalize(glm::cross(camDevant, camera_up)) * cameraSpeed;
	}
	if(io.KeysDown[GLFW_KEY_Q])
	{
		if (camera_pos.y > 0.5f)
			camera_pos.y -= 2.0f;
	}
	if(io.KeysDown[GLFW_KEY_E])
	{
		camera_pos.y += 2.0f;
	}

	if (camera_pos.z <= -1000.0)
	{
		camera_pos.z = -1000.0;
	}
	if (camera_pos.z >= 1000.0)
	{
		camera_pos.z = 1000.0;
	}
	if (camera_pos.x <= -1000.0)
	{
		camera_pos.x = -1000.0;
	}
	if (camera_pos.x >= 1000.0)
	{
		camera_pos.x = 1000.0;
	}
}

void FPSCamera::mouse_move(int x, int y)
{
	ImGuiIO& io = ImGui::GetIO();
	ImVec2 mp = io.MousePos;
	if(mp.x == -FLT_MAX && mp.y == -FLT_MAX)
	{
		return;
	}
	x = mp.x;
	y = mp.y;

	if (wrap_mouse) {
		wrap_mouse = false;
		last_x = x;
		last_y = y;
	}
	else {
		if (first_mouse) {
			last_x = x;
			last_y = y;
			first_mouse = false;
		}

		GLfloat xoffset = x - last_x;
		GLfloat yoffset = last_y - y;
		last_x = x;
		last_y = y;

		GLfloat sensitivity = 0.3;
		xoffset *= sensitivity;
		yoffset *= sensitivity;

		yaw += xoffset;
		pitch += yoffset;

		if (pitch > 89.0f) pitch = 89.0f;
		if (pitch < -89.0f) pitch = -89.0f;

		glm::vec3 front;
		front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
		front.y = sin(glm::radians(pitch));
		front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
		camera_front = glm::normalize(front);

		wrap_mouse = true;
		//glutWarpPointer(glutGet(GLUT_WINDOW_WIDTH) / 2, glutGet(GLUT_WINDOW_HEIGHT) / 2);
	}
}

void FPSCamera::keyboard_press(unsigned char btn, int x, int y)
{
	switch(btn)
	{
	case 'a':
		btns[0] = true;
		break;
	case 'w':
		btns[1] = true;
		break;
	case 's':
		btns[2] = true;
		break;
	case 'd':
		btns[3] = true;
		break;
	case 'q':
		btns[4] = true;
		break;
	case 'e':
		btns[5] = true;
		break;
	case 'r':
		btns[6] = true;
		break;
	case 'i':
		btns[7] = true;
		break;
	case 'k':
		btns[8] = true;
		break;
	case 'j':
		btns[9] = true;
		break;
	case 'l':
		btns[10] = true;
		break;
	case 27:
		//glutLeaveMainLoop();
		break;

	}
}

void FPSCamera::keyboard_release(unsigned char btn, int x, int y)
{
	switch (btn)
	{
	case 'a':
		btns[0] = false;
		break;
	case 'w':
		btns[1] = false;
		break;
	case 's':
		btns[2] = false;
		break;
	case 'd':
		btns[3] = false;
		break;
	case 'q':
		btns[4] = false;
		break;
	case 'e':
		btns[5] = false;
		break;
	case 'r':
		btns[6] = false;
		break;
	case 'i':
		btns[7] = false;
		break;
	case 'k':
		btns[8] = false;
		break;
	case 'j':
		btns[9] = false;
		break;
	case 'l':
		btns[10] = false;
		break;

	}
}