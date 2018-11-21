#include "camera.h"

#include <GL/freeglut.h>

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

	if (btns[0] == true) {
		camera_pos -= glm::normalize(glm::cross(camDevant, camera_up)) * cameraSpeed;
	}
	if (btns[1] == true) {
		camera_pos += cameraSpeed * camDevant;
	}
	if (btns[2] == true) {
		camera_pos -= cameraSpeed * camDevant;
	}
	if (btns[3] == true) {
		camera_pos += glm::normalize(glm::cross(camDevant, camera_up)) * cameraSpeed;
	}
	if(btns[4] == true)
	{
		if (camera_pos.y > 0.5f)
			camera_pos.y -= 2.0f;
	}
	if(btns[5] == true)
	{
		camera_pos.y += 2.0f;
	}
	if (btns[6] == true) {

	}
	if (camera_pos.z <= -200.0)
	{
		camera_pos.z = -200.0;
	}
	if (camera_pos.z >= 200.0)
	{
		camera_pos.z = 200.0;
	}
	if (camera_pos.x <= -200.0)
	{
		camera_pos.x = -200.0;
	}
	if (camera_pos.x >= 200.0)
	{
		camera_pos.x = 200.0;
	}
}

void FPSCamera::mouse_move(int x, int y)
{
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
		glutWarpPointer(glutGet(GLUT_WINDOW_WIDTH) / 2, glutGet(GLUT_WINDOW_HEIGHT) / 2);
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

	case 27:
		glutLeaveMainLoop();
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
	}
}