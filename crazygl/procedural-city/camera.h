#ifndef CAMERA_H
#define CAMERA_H
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <GL/glew.h>

class FPSCamera
{
private:
	glm::vec3 camera_pos = glm::vec3(0.0f, 0.5f, 7.0f);
	glm::vec3 camera_front = glm::vec3(0.0f, 0.25f, -1.0f);
	glm::vec3 camera_up = glm::vec3(0.0f, 5.0f, 0.0f);

	float	yaw = -90.0f;
	float	pitch = 0.0f;

	float	fov = 70.0f;

	float	last_x = 0;
	float	last_y = 0;

	bool	wrap_mouse = true;
	bool	first_mouse = 0;

	bool	btns[11] { false,false,false,false,false,false,false,false,false,false,false};

	float	tx = 1.0f;
	float	ty = 1.0f;
	float	tz = 1.0f;

public:
	FPSCamera();

	glm::mat4 getLookAt();
	glm::mat4 getView();

	void update();

	void mouse_move(int x, int y);
	void keyboard_press(unsigned char btn, int x, int y);
	void keyboard_release(unsigned char btn, int x, int y);

	void addTX(float tx)
	{
		this->tx += tx;
	}

	void addTY(float ty)
	{
		this->ty += ty;
	}

	glm::vec3 getT() const
	{
		return { tx,ty,tz };
	}

	void addFOV(float v)
	{
		fov += v;
	}

	float getFOV()
	{
		return fov;
	}
};



#endif 
